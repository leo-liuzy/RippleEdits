from src.query import Query, LookupQuery


class TestCase:

    OR_TEST_CONDITION = 'OR'
    AND_TEST_CONDITION = 'AND'

    def __init__(self, test_query, condition_queries=None, test_condition=OR_TEST_CONDITION):
        if condition_queries is None:
            condition_queries = []
        if type(test_query) is list:
            self._test_queries = test_query
        else:
            self._test_queries = [test_query]
        self._condition_queries = condition_queries
        self._test_condition = test_condition

    def get_test_queries(self):
        return self._test_queries

    def get_test_condition(self):
        return self._test_condition

    def get_condition_queries(self):
        return self._condition_queries

    def to_dict(self):
        return {
            'test_queries': [query.to_dict() for query in self._test_queries],
            'test_condition': self._test_condition,
            'condition_queries': [query.to_dict() for query in self._condition_queries]
        }

    @staticmethod
    def from_dict(d):
        tests = [Query.from_dict(test) for test in d['test_queries']]
        test_condition = d['test_condition']
        conditions = [Query.from_dict(condition) for condition in d['condition_queries']]
        return TestCase(tests, conditions, test_condition)

    def __str__(self):
        res = 'Test Queries:\n'
        for query in self._test_queries:
            query_dict = query.to_dict()
            res += f"Query: {query_dict['prompt']}, " \
                   f"Answer: {query_dict['answers'][0]['value']}\n"
        res += f'Test Condition: {self._test_condition}\n'
        res += 'Condition Queries:\n'
        for query in self._condition_queries:
            query_dict = query.to_dict()
            res += f"Query: {query_dict['prompt']}, " \
                   f"Answer: {query_dict['answers'][0]['value']}\n"
        return res

class LookupTestCase(TestCase):
    @staticmethod
    def from_dict(d, edited_fact = None, filter_mode: str = 'verbatim'):
        
        tests = [LookupQuery.from_dict(test) for test in d['test_queries'] if len([y for x in test["answers"] for y in ([x["value"]] + x["aliases"]) if y != ""]) > 0]
        # import pdb; pdb.set_trace()
        filter_tests = []
        for test in tests:
            answers = test.get_lookup_answers()
            assert len(answers) == 1, f"Filter test should have only one answer. Found: {answers}"
            answers = answers[0]
            if filter_mode == 'verbatim':
                if answers[0] in edited_fact:
                    # if the answer is in the edited input, we want to keep it
                    filter_tests.append(test)
            elif filter_mode == 'non-verbtaim':
                if answers[0] not in edited_fact:
                    # if the answer is not in the edited input, we want to keep it
                    filter_tests.append(test)
            else: 
                assert filter_mode == 'all', f"Filter mode should be 'verbatim', 'not_verbatim' or 'all'. Found: {filter_mode}"
                # if the answer is in the edited input, we want to keep it
                filter_tests.append(test)
        # import pdb; pdb.set_trace()
        if len(filter_tests) == 0:
            return 
        tests = filter_tests
        if len(tests) != len(d['test_queries']):
            print(f"Warning: Some test queries have no answers. Skipping {len(d['test_queries']) - len(tests)} queries.")
        test_condition = d['test_condition']
        conditions = [LookupQuery.from_dict(condition) for condition in d['condition_queries']]
        return TestCase(tests, conditions, test_condition)