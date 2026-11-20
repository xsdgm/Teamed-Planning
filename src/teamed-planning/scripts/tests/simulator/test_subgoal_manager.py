"""Periodic direction selection from delivered roads and vehicle broadcasts."""
import contextlib
import io
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from common.graph_structure import TopologicalGraph
from ground.ground_main import GroundRobotController
from ground.subgoal_manager import SubgoalManager
from simulator.mappo_experiment import Experiment


def graph_with(*positions):
    graph=TopologicalGraph()
    for i,position in enumerate(positions):
        graph.add_node(f'n{i}',tuple(position))
    return graph


def side_road():
    # A confirmed road 30 degrees left of the direct eastward heading.
    origin = np.array((100., 500.))
    direction = np.array((np.cos(np.pi / 6), np.sin(np.pi / 6)))
    graph = graph_with(*(origin + distance * direction for distance in (0, 45, 90, 135, 180)))
    for i in range(4):
        graph.add_edge(f'n{i}', f'n{i+1}')
    return graph


def road_from_origin(bearing_degrees, length, east_branch=0.):
    angle = np.deg2rad(bearing_degrees)
    end = length * np.array([np.cos(angle), np.sin(angle)])
    graph = graph_with((0., 0.), end)
    graph.add_edge('n0', 'n1')
    if east_branch:
        graph.add_node('east', (east_branch, 0.))
        graph.add_edge('n0', 'east')
    return graph


class SubgoalManagerTests(unittest.TestCase):
    def test_far_goal_gets_a_subgoal_on_the_straight_line(self):
        manager=SubgoalManager((800.,500.),radius=200.)
        target,changed=manager.update((100.,500.),None,0.)
        self.assertTrue(changed)
        np.testing.assert_allclose(target,(300.,500.),atol=1e-6)
        self.assertFalse(manager.final)

    def test_review_rolls_direction_forward_without_arriving(self):
        manager = SubgoalManager((800., 500.), review_interval=7.)
        first, _ = manager.update((100., 500.), None, 100.)
        position = (140., 500.)
        self.assertGreater(np.linalg.norm(first - position), manager.arrival)
        target, changed = manager.update(position, None, 106.9)
        self.assertFalse(changed)
        np.testing.assert_allclose(target, first)
        target, changed = manager.update(position, None, 107.)
        self.assertTrue(changed)
        self.assertGreater(target[0], first[0])
        self.assertAlmostEqual(target[1], first[1])
        self.assertFalse(manager.direction_changed)
        self.assertEqual(manager.arrivals, 0)
        self.assertEqual(manager.failed, 0)

    def test_goal_within_radius_is_the_final_target(self):
        manager=SubgoalManager((250.,500.),radius=200.)
        target,_=manager.update((100.,500.),None,0.)
        np.testing.assert_allclose(target,(250.,500.))
        self.assertTrue(manager.final)

    def test_small_changes_in_goal_bearing_keep_the_current_direction(self):
        manager = SubgoalManager((1000., 0.))
        manager.update((0., 0.), None, 0.)
        # The goal bearing drifts through 4 and 8 degrees. Keeping the
        # eastward heading costs less than 3 m of remaining distance.
        for offset, now in ((70., 15.), (140., 30.)):
            target, _ = manager.update((0., -offset), None, now)
            self.assertFalse(manager.direction_changed)
            np.testing.assert_allclose(target, (200., -offset), atol=1e-6)
        target, _ = manager.update((0., -500.), None, 45.)
        self.assertTrue(manager.direction_changed)
        self.assertGreater(target[1], -500.)

    def test_repeated_marginal_side_road_advantage_does_not_turn_the_car(self):
        manager = SubgoalManager((700., 0.))
        direct, _ = manager.update((0., 0.), None, 0.)
        # Both routes are confirmed. The 15-degree branch extends 41 m
        # sideways beyond the east axis, with a score advantage of only 7.
        road = road_from_origin(15., 160., east_branch=149.)
        for now in (15., 30., 45.):
            target, changed = manager.update((0., 0.), road, now)
            self.assertFalse(changed)
            np.testing.assert_array_equal(target, direct)
        self.assertEqual(manager.reviews, 4)

    def test_sharp_turn_requires_a_larger_improvement(self):
        # Each road offers about 15 units over continuing east. The mild
        # turn is worth taking, the right-angle turn needs a longer road.
        for degrees, length, east, should_turn in ((15., 160., 144., True), (90., 162., 0., False)):
            with self.subTest(degrees=degrees):
                manager = SubgoalManager((700., 0.))
                direct, _ = manager.update((0., 0.), None, 0.)
                road = road_from_origin(degrees, length, east_branch=east)
                target, changed = manager.update((0., 0.), road, 15.)
                self.assertEqual(changed, should_turn)
                if should_turn:
                    self.assertAlmostEqual(np.rad2deg(np.arctan2(target[1], target[0])), degrees)
                else:
                    np.testing.assert_array_equal(target, direct)
                    longer = road_from_origin(degrees, 175.)
                    target, changed = manager.update((0., 0.), longer, 30.)
                    self.assertTrue(changed)
                    np.testing.assert_allclose(target, (0., 200.), atol=1e-6)

    def test_mission_goal_requires_a_worthwhile_turn_until_final_approach(self):
        manager = SubgoalManager((300., 0.))
        manager.update((0., 0.), None, 0.)
        # The mission goal entered the 200 m radius only 3 m off the old
        # course. Roll east, then take over at the 25 m finishing distance.
        position = np.array((110., 3.))
        target, _ = manager.update(position, None, 15.)
        self.assertFalse(manager.final)
        self.assertFalse(manager.direction_changed)
        np.testing.assert_allclose(target, position + (np.linalg.norm(manager.goal - position), 0.))
        target, _ = manager.update((282., 3.), None, 16.)
        self.assertTrue(manager.final)
        np.testing.assert_array_equal(target, manager.goal)

    def test_already_selected_mission_goal_stays_fixed_when_car_moves(self):
        manager = SubgoalManager((180., 0.))
        manager.update((0., 0.), None, 0.)
        target, changed = manager.update((0., -3.), None, 15.)
        self.assertFalse(changed)
        self.assertTrue(manager.final)
        np.testing.assert_array_equal(target, manager.goal)

    def test_arrival_selects_the_next_subgoal(self):
        manager=SubgoalManager((800.,500.),radius=200.)
        manager.update((100.,500.),None,0.)
        target,changed=manager.update((280.,500.),None,180.)
        self.assertTrue(changed)
        np.testing.assert_allclose(target,(480.,500.),atol=1e-6)
        self.assertEqual(manager.arrivals,1)

    def test_stall_temporarily_changes_the_approach_from_the_same_position(self):
        manager=SubgoalManager((800.,500.),radius=200.,stall=100.)
        first,_=manager.update((100.,500.),None,0.)
        _,changed=manager.update((100.,500.),None,99.)
        self.assertFalse(changed)
        second,changed=manager.update((100.,500.),None,100.)
        self.assertTrue(changed)
        self.assertEqual(manager.failed,1)
        self.assertFalse(np.allclose(second,first))
        # An adjacent direction stays usable even with its marker inside the
        # former 60 m disk around the stalled marker.
        self.assertLess(np.linalg.norm(second - first), 60.)
        self.assertGreater(second[0],100.)
        repeated, _ = manager.update((100., 500.), None, 115.)
        np.testing.assert_allclose(repeated, second)

    def test_failure_penalty_decays_so_an_approach_can_be_retried(self):
        manager=SubgoalManager((800.,500.),radius=200.,stall=100.,penalty_decay=50.)
        first,_=manager.update((100.,500.),None,0.)
        manager.update((100.,500.),None,100.)
        retried,_=manager.update((100.,500.),None,2000.)
        np.testing.assert_allclose(retried,first,atol=1e-6)

    def test_failed_forward_marker_remains_available_from_a_different_entrance(self):
        manager = SubgoalManager((500., 500.), stall=100.)
        marker, _ = manager.update((100., 500.), None, 0.)
        np.testing.assert_allclose(marker, (300., 500.))
        detour, _ = manager.update((100., 500.), None, 100.)
        self.assertFalse(np.allclose(detour, marker))

        # At the southern entrance a mapped road runs north toward the very
        # same marker. A failure on the western approach cannot block it.
        entrance = (300., 300.)
        road = graph_with(entrance, (300., 400.), marker)
        road.add_edge('n0', 'n1')
        road.add_edge('n1', 'n2')
        target, _ = manager.update(entrance, road, 115.)
        np.testing.assert_allclose(target, marker, atol=1e-6)
        self.assertEqual(manager.failed, 1)

    def test_moving_to_an_adjacent_lane_releases_the_old_approach(self):
        manager = SubgoalManager((800., 500.), stall=100.)
        previous, _ = manager.update((100., 500.), None, 0.)
        manager.update((100., 500.), None, 100.)
        self.assertEqual(manager.failed, 1)

        # The new approach is nearly parallel, and its marker is only 7 m
        # from the old one. Only the local entrance has materially changed.
        new_car = np.array((100., 510.))
        direct = manager.goal - new_car
        lane = graph_with(new_car, new_car + 160. * direct / np.linalg.norm(direct))
        lane.add_edge('n0', 'n1')
        target, _ = manager.update(new_car, lane, 115.)
        np.testing.assert_allclose(target, new_car + manager.radius * direct / np.linalg.norm(direct))
        self.assertLess(np.linalg.norm(target - previous), 10.)

    def test_newly_connected_road_can_restore_the_failed_bearing(self):
        manager = SubgoalManager((800., 500.), stall=100.)
        car = (100., 500.)
        road = graph_with(car, (115., 500.))
        road.add_edge('n0', 'n1')
        original, _ = manager.update(car, road, 0.)
        detour, _ = manager.update(car, road, 100.)
        self.assertEqual(manager.failed, 1)
        self.assertFalse(np.allclose(detour, original))
        same_evidence, _ = manager.update(car, road, 115.)
        self.assertFalse(np.allclose(same_evidence, original))

        # New edges show a route bending around the obstruction toward the
        # original bearing. The direct marker has stayed at the same coordinate.
        road.add_node('turn', (125., 506.))
        road.add_node('ahead', (250., 500.))
        road.add_edge('n1', 'turn')
        road.add_edge('turn', 'ahead')
        target, _ = manager.update(car, road, 130.)
        np.testing.assert_allclose(target, original, atol=1e-6)
        self.assertEqual(manager.failed, 1)

    def test_repeated_dead_ends_back_off_toward_the_rear(self):
        manager=SubgoalManager((800.,500.),radius=200.,stall=10.,penalty_radius=120.)
        targets=[manager.update((100.,500.),None,0.)[0]]
        for i in range(1,40):
            target,changed=manager.update((100.,500.),None,10.*i)
            if changed:
                targets.append(target)
        self.assertTrue(any(t[0]<100. for t in targets))

    def test_delivered_road_can_turn_the_heading_on_the_next_review(self):
        manager = SubgoalManager((800., 500.))
        car = (100., 500.)
        direct, _ = manager.update(car, None, 0.)
        graph = side_road()
        target, changed = manager.update(car, graph, 14.9)
        self.assertFalse(changed)
        np.testing.assert_array_equal(target, direct)
        target, changed = manager.update(car, graph, 15.)
        self.assertTrue(changed)
        self.assertTrue(manager.direction_changed)
        self.assertGreater(target[0], car[0])
        self.assertGreater(target[1], car[1] + 10.)
        self.assertEqual((manager.arrivals, manager.failed), (0, 0))
        repeated, changed = manager.update(car, graph, 30.)
        self.assertFalse(changed)
        np.testing.assert_allclose(repeated, target)

    def test_disconnected_road_or_guessed_bridge_cannot_redirect_the_car(self):
        for bridged in (False, True):
            with self.subTest(bridged=bridged):
                graph = side_road()
                graph.graph.remove_edge('n0', 'n1')
                if bridged:
                    graph.add_edge('n0', 'n1', attributes={'bridged': True})
                manager = SubgoalManager((800., 500.))
                direct, _ = manager.update((100., 500.), None, 0.)
                target, changed = manager.update((100., 500.), graph, 15.)
                self.assertFalse(changed)
                np.testing.assert_allclose(target, direct)

    def test_reported_road_extending_beyond_lookahead_still_informs_heading(self):
        car = np.array((100., 500.))
        road_end = car + 400. * np.array((np.cos(np.pi / 6), np.sin(np.pi / 6)))
        road = graph_with(car, road_end)
        road.add_edge('n0', 'n1')
        manager = SubgoalManager((800., 500.))
        target, _ = manager.update(car, road, 0.)
        np.testing.assert_allclose(target - car, (road_end - car) / 2.)

    def test_continuous_cross_road_can_outweigh_a_geometric_shortcut(self):
        manager = SubgoalManager((350., 0.))
        car = (0., 0.)
        direct, _ = manager.update(car, None, 0.)
        np.testing.assert_allclose(direct, (200., 0.))
        road = graph_with(car, (0., 100.), (0., 200.))
        road.add_edge('n0', 'n1')
        road.add_edge('n1', 'n2')
        # The north marker is 253 m farther from the goal than the east
        # marker. Its continuous 200 m of roadway earns 300 m of credit.
        target, _ = manager.update(car, road, 15.)
        np.testing.assert_allclose(target, (0., 200.), atol=1e-6)

    def test_a_far_endpoint_after_leaving_the_corridor_does_not_dominate_a_continuous_road(self):
        manager = SubgoalManager((500., 0.))
        car = np.array((0., 0.))
        ray = np.array([[1., 0.]])
        lengths = np.array([200.])
        road = graph_with(car, (20., 0.), (40., 35.), (145., 0.))
        for index in range(3):
            road.add_edge(f'n{index}', f'n{index + 1}')
        bent_support, _ = manager._road_support(car, road, ray, lengths, None)
        road = graph_with(car, (80., 8.))
        road.add_edge('n0', 'n1')
        continuous_support, _ = manager._road_support(car, road, ray, lengths, None)
        self.assertGreater(continuous_support[0], bent_support[0])

    def test_nearby_isolated_node_does_not_capture_heading_between_reviews(self):
        manager = SubgoalManager((800., 500.))
        direct, _ = manager.update((100., 500.), None, 0.)
        target, changed = manager.update((260., 500.), graph_with((305., 510.)), 2.)
        self.assertFalse(changed)
        np.testing.assert_allclose(target, direct)

    def test_new_connected_road_support_postpones_stall_only_until_max_wait(self):
        manager = SubgoalManager((800., 500.), review_interval=5., stall=20., max_wait=45.)
        car = (100., 500.)
        manager.update(car, None, 0.)
        for now in (10., 20., 30., 40., 45.):
            graph = graph_with(car, (100. + 2 * now, 500.))
            graph.add_edge('n0', 'n1')
            manager.update(car, graph, now)
            self.assertEqual(manager.failed, int(now == 45.))

    def test_graph_growth_outside_the_corridor_is_not_progress(self):
        manager=SubgoalManager((800.,500.),radius=200.,stall=100.)
        manager.update((100.,500.),None,0.)
        nodes=[]
        for t in range(1,11):
            nodes+=[(50.+t,300.+i) for i in range(11)]
            manager.update((100.,500.),graph_with(*nodes),10.*t)
        self.assertEqual(manager.failed,1)

    def test_driving_away_from_the_subgoal_fails_it(self):
        manager=SubgoalManager((800.,500.),radius=200.)
        manager.update((100.,500.),None,0.)
        manager.update((150.,500.),None,50.)
        manager.update((110.,500.),None,90.)
        self.assertEqual(manager.failed, 0)  # Exactly 40 m back along the heading.
        second,changed=manager.update((105.,500.),None,95.)
        self.assertTrue(changed)
        self.assertEqual(manager.failed,1)
        self.assertFalse(np.isclose(second[1], 500.))

    def test_subgoals_stay_inside_the_surveyed_bounds(self):
        manager=SubgoalManager((990.,990.),radius=200.,stall=1.,world_bounds=(0.,0.,1000.,1000.))
        for i in range(30):
            target,_=manager.update((900.,500.),None,float(i))
            self.assertTrue(((target>=10.)&(target<=990.)).all())

    def test_near_boundary_uses_a_forward_direction_with_meaningful_lookahead(self):
        manager = SubgoalManager((995., 900.), world_bounds=(0., 0., 1000., 1000.))
        car = np.array((989.9, 500.))
        target, _ = manager.update(car, None, 0.)
        self.assertLessEqual(target[0], 990.)
        self.assertGreater(target[1], car[1])
        self.assertGreaterEqual(np.linalg.norm(target - car), manager.arrival)


class SubgoalVehicleTests(unittest.TestCase):
    def test_vehicle_follows_bypass_and_redirects_fleet_only_when_road_leaves_corridor(self):
        road = graph_with((0., 0.), (20., 0.), (25., 6.), (50., 17.), (80., 27.))
        for i in range(4):
            road.add_edge(f'n{i}', f'n{i + 1}')
        obs = dict(self=dict(position=[0., 0., 0.]), neighbors=[], map_resolution=.5,
                   occupancy_map=np.zeros((16, 16), np.uint8), full_state=dict(time=0.))
        with contextlib.redirect_stdout(io.StringIO()):
            car = GroundRobotController('test', subgoal_radius=200., subgoal_review_interval=5.,
                                        fusion_interval=5.)
            car.set_goal((700., 0.))
            east = car.get_action(obs)['broadcast']['target']

            def deliver_road(now):
                obs['full_state']['time'] = now
                obs['neighbors'] = [dict(type='drone', broadcast=dict(
                    drone_id='drone_0', position=[0., 0., 15.], graph=road.to_message()))]
                return car.get_action(obs)['broadcast']

            # The ground path bends while the fleet still receives the eastward heading.
            local = deliver_road(5.)
            self.assertIn((50., 17.), local['planned_path'])
            np.testing.assert_array_equal(local['target'], east)
            self.assertFalse(car.subgoals.direction_changed)

            # New reports extend the same passage beyond the 30 m corridor.
            road.add_node('extension', (150., 50.))
            road.add_edge('n4', 'extension')
            extended = deliver_road(10.)
            self.assertIn((150., 50.), extended['planned_path'])
            self.assertTrue(car.subgoals.direction_changed)
            x, y = extended['target']
            self.assertAlmostEqual(np.rad2deg(np.arctan2(y, x)), 15.)

    def test_forward_reviews_keep_the_road_prefix_and_extend_before_its_end(self):
        road = TopologicalGraph('drone_0')
        for x in range(100, 231, 5):
            road.add_node(str(x), (x, 500.), dict(terrain_type='TRAVERSABLE', observation_count=20))
            if x > 100:
                road.add_edge(str(x - 5), str(x))
        packet = dict(type='drone', broadcast=dict(
            drone_id='drone_0', position=[100., 500., 15.], graph=road.to_message()))
        obs = dict(neighbors=[packet], map_resolution=.5, occupancy_map=np.zeros((16, 16), np.uint8))
        with contextlib.redirect_stdout(io.StringIO()):
            car = GroundRobotController('test', subgoal_radius=60., subgoal_arrival=5.,
                                        subgoal_review_interval=5.)
            car.set_goal((800., 500.))
            planner = car.global_planner
            for step in range(61):
                x = 100. + step
                obs.update(self=dict(position=[x, 500., 0.]), full_state=dict(time=float(step)))
                action = car.get_action(obs)
                obs['neighbors'] = []
                self.assertTrue(planner.using_graph)
                self.assertGreater(action['motion'][0], 0.)
                if step == 0:
                    first_path = planner.current_path
                    old_end = first_path[-1][0]
                if step == 5:
                    self.assertGreater(action['broadcast']['target'][0], old_end)
                    self.assertIs(planner.current_path, first_path)
                    self.assertGreater(planner.current_waypoint_idx, 1)
                if planner.current_path[-1][0] > old_end:
                    self.assertLess(x, old_end)
                    np.testing.assert_allclose(planner.goal_position, action['broadcast']['target'])
                    np.testing.assert_allclose(planner.current_path[-1], planner.goal_position)
                    break
            else:
                self.fail('The vehicle reached the old marker before extending the route')

    def test_review_broadcast_uses_the_graph_fused_and_filtered_on_that_frame(self):
        for blocked in (False, True):
            with self.subTest(blocked=blocked), contextlib.redirect_stdout(io.StringIO()):
                obs = dict(self=dict(position=np.array([100., 500., 0.]), orientation=0.),
                           neighbors=[], occupancy_map=np.zeros((16, 16), np.uint8),
                           map_resolution=.5, time_step=1., full_state=dict(time=0.))
                car = GroundRobotController('test', subgoal_radius=200., subgoal_review_interval=7.)
                car.set_goal((800., 500.))
                first = car.get_action(obs)['broadcast']['target']
                np.testing.assert_allclose(first, (300., 500.))
                obs['neighbors'] = [dict(type='drone', broadcast=dict(
                    drone_id='drone_0', position=[100., 500., 15.], graph=side_road().to_message()))]
                obs['full_state']['time'] = 7.
                if blocked:
                    obs['occupancy_map'][:, 12] = 1  # Wall across the road's first edge.
                target = car.get_action(obs)['broadcast']['target']
                self.assertTrue(car.current_drone_communications['drone_0']['fused'])
                if blocked:
                    np.testing.assert_allclose(target, first)
                else:
                    self.assertGreater(target[1], first[1] + 10.)
                np.testing.assert_allclose(target, car.goal_position)
                self.assertEqual(car.final_goal, (800., 500.))

    def test_controller_relay_mode_drives_to_and_broadcasts_subgoals(self):
        # The ROS ground node uses this controller path with ~subgoal_radius.
        obs=dict(self=dict(position=np.array([100.,500.,0.]),orientation=0.),neighbors=[],
                 occupancy_map=np.zeros((16,16),np.uint8),map_resolution=.5,time_step=1.,
                 full_state=dict(time=0.))
        with contextlib.redirect_stdout(io.StringIO()):
            car=GroundRobotController('test',resolution=.5,subgoal_radius=200.,world_bounds=(0.,0.,1000.,1000.))
            car.set_goal((800.,500.))
            self.assertIsNone(car.goal_position)
            action=car.get_action(obs)
            np.testing.assert_allclose(car.goal_position,(300.,500.),atol=1e-6)
            np.testing.assert_allclose(action['broadcast']['target'],(300.,500.),atol=1e-6)
            self.assertEqual(car.final_goal,(800.,500.))
            car.stop()
            self.assertIsNone(car.goal_position or car.subgoals)
            self.assertIsNone(car.get_action(obs)['broadcast']['target'])
            car.set_goal((800.,500.));car.reset()
            self.assertIsNone(car.subgoals)
            plain=GroundRobotController('test',resolution=.5)
            plain.set_goal((800.,500.))
            self.assertEqual(plain.goal_position,(800.,500.))
            self.assertIsNone(plain.subgoals)

    def test_vehicle_drives_to_and_broadcasts_the_subgoal(self):
        torch.set_num_threads(1)
        with contextlib.redirect_stdout(io.StringIO()):
            env=Experiment(20000,None,subgoal_radius=30.,subgoal_review_interval=2.)
            self.assertEqual(env.subgoals.review_interval, 2.)
            self.assertGreater(np.linalg.norm(env.goal-env.start),30.)
            targets=[env.subgoals.target.copy()]
            for frame in range(6):
                env.step(frame)
                targets.append(env.subgoals.target.copy())
            self.assertFalse(env.subgoals.final)
            np.testing.assert_allclose(env.ground_controller.goal_position,targets[-1])
            np.testing.assert_allclose(env.ground_action['broadcast']['target'],targets[-1])
            # Rank 0 hears the vehicle directly (a step or two late) and plans
            # toward a subgoal, never toward the final goal.
            heard=np.asarray(env.controllers[0].navigator.ground_robot_target)
            self.assertTrue(any(np.allclose(heard,t) for t in targets))
            self.assertIn('subgoal_failures',env.summary())
            env.env.close()


if __name__=='__main__':
    unittest.main()
