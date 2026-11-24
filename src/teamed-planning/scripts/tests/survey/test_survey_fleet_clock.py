import unittest

from survey.survey_fleet_runtime import _survey_reason


class FleetSurveyClockTests(unittest.TestCase):
    def test_minimum_hold_and_timeout_use_the_last_evaluation(self):
        active_request = {'issued_at_s': 0}
        self.assertIsNone(_survey_reason(15, active_request, 0, True, 20, 60))
        self.assertEqual(_survey_reason(20, active_request, 0, True, 20, 60), 'evidence')
        # The selector retained the old physical request at t=20. It cannot
        # immediately trigger a second review at t=25 on the existing answer.
        self.assertIsNone(_survey_reason(25, active_request, 20, True, 20, 60))
        self.assertIsNone(_survey_reason(75, active_request, 20, False, 20, 60))
        self.assertEqual(_survey_reason(80, active_request, 20, False, 20, 60), 'timeout')
        self.assertEqual(active_request['issued_at_s'], 0)

    def test_empty_candidate_set_can_be_reconsidered_on_next_check(self):
        self.assertEqual(_survey_reason(0, None, None, False, 20, 60), 'idle')
        self.assertEqual(_survey_reason(5, None, 0, False, 20, 60), 'idle')


if __name__ == '__main__':
    unittest.main()
