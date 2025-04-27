import random
from enum import Enum, auto
import json
from pathlib import Path

from fact import Fact, LookupFact
from testcase import TestCase, LookupTestCase


class TestsAxis(Enum):
    LOGICAL_CONSTRAINTS = auto()
    TWO_HOP = auto()
    FORWARD_TWO_HOP = auto()
    SUBJECT_PARAPHRASING = auto()
    MAKING_UP = auto()
    PREVIOUS_STORAGE = auto()


class Example:

    def __init__(self,
                 fact: Fact,
                 making_up_tests: list = [],
                 logical_constraints: list = [],
                 subject_paraphrasing_tests: list = [],
                 two_hop_tests: list = [],
                 forward_two_hop_tests: list = [],
                 prev_storage_tests: list = []):
        self.fact = fact
        self.making_up_tests = making_up_tests
        self.logical_constraints = logical_constraints
        self.subject_paraphrasing_tests = subject_paraphrasing_tests
        self.two_hop_tests = two_hop_tests
        self.forward_two_hop_tests = forward_two_hop_tests
        self.prev_storage_tests = prev_storage_tests

    def create_example_dict(self, example_type):
        return {
            'example_type': example_type,
            'edit': self.fact.to_dict(),
            'Relation_Specificity': [test.to_dict() for test in self.making_up_tests],
            'Logical_Generalization': [test.to_dict() for test in self.logical_constraints],
            'Subject_Aliasing': [test.to_dict() for test in self.subject_paraphrasing_tests],
            'Compositionality_I': [test.to_dict() for test in self.two_hop_tests],
            'Compositionality_II': [test.to_dict() for test in self.forward_two_hop_tests],
            'Forgetfulness': [test.to_dict() for test in self.prev_storage_tests],
        }

    @staticmethod
    def from_dict(d):
        fact = Fact.from_dict(d['edit'])
        making_up_tests = [TestCase.from_dict(test) for test in d['Relation_Specificity']]
        logical_constraints = [TestCase.from_dict(test) for test in d['Logical_Generalization']]
        subject_paraphrasing_tests = [TestCase.from_dict(test) for test in d['Subject_Aliasing']]
        two_hop_tests = [TestCase.from_dict(test) for test in d['Compositionality_I']]
        forward_two_hop_tests = [TestCase.from_dict(test) for test in d['Compositionality_II']]
        prev_storage_tests = [TestCase.from_dict(test) for test in d['Forgetfulness']]
        if d['example_type'] in ['random', 'popular']:
            previous_fact = Fact.from_dict(d['edit']['original_fact'])
            return CounterFactualExample(fact, previous_fact, making_up_tests, logical_constraints,
                                         subject_paraphrasing_tests, two_hop_tests, forward_two_hop_tests, prev_storage_tests)
        elif d['example_type'] == 'recent':
            return RecentlyAddedExample(fact, making_up_tests, logical_constraints, subject_paraphrasing_tests,
                                        two_hop_tests, forward_two_hop_tests, prev_storage_tests)
        else:
            print('Unknown fact type')

    def __str__(self):
        res = f'Fact: {str(self.fact)}\n'
        res += f'Making Up tests:\n'
        res += self.str_list_of_tests(self.making_up_tests)
        res += '\n'
        res += f'Logical Constraints:\n'
        res += self.str_list_of_tests(self.logical_constraints)
        res += '\n'
        res += f'Subject Paraphrasing tests:\n'
        res += self.str_list_of_tests(self.subject_paraphrasing_tests)
        res += '\n'
        res += f'Two-Hop tests:\n'
        res += self.str_list_of_tests(self.two_hop_tests)
        res += '\n'
        res += f'Forward Two-Hop tests:\n'
        res += self.str_list_of_tests(self.forward_two_hop_tests)
        res += '\n'
        res += f'Previous Storage tests:'
        res += self.str_list_of_tests(self.prev_storage_tests)
        res += '\n'
        return res
    
    @staticmethod
    def str_list_of_tests(tests: list):
        res = ''
        for test in tests:
            res += f'{str(test)}\n'
        return res

class LookupExample(Example):
    def __init__(self,
                 fact: Fact,
                 making_up_tests: list = [],
                 logical_constraints: list = [],
                 subject_paraphrasing_tests: list = [],
                 two_hop_tests: list = [],
                 forward_two_hop_tests: list = [],
                 prev_storage_tests: list = []):
        self.fact = fact
        self.making_up_tests = making_up_tests
        self.logical_constraints = logical_constraints
        self.subject_paraphrasing_tests = subject_paraphrasing_tests
        self.two_hop_tests = two_hop_tests
        self.forward_two_hop_tests = forward_two_hop_tests
        self.prev_storage_tests = prev_storage_tests

    def create_example_dict(self, example_type):
        return {
            'example_type': example_type,
            'edit': self.fact.to_dict(),
            'Relation_Specificity': [test.to_dict() for test in self.making_up_tests],
            'Logical_Generalization': [test.to_dict() for test in self.logical_constraints],
            'Subject_Aliasing': [test.to_dict() for test in self.subject_paraphrasing_tests],
            'Compositionality_I': [test.to_dict() for test in self.two_hop_tests],
            'Compositionality_II': [test.to_dict() for test in self.forward_two_hop_tests],
            'Forgetfulness': [test.to_dict() for test in self.prev_storage_tests],
        }

    @staticmethod
    def from_dict(d, filter_list: list = [], filter_mode: str = 'verbatim'):
        
        fact = LookupFact.from_dict(d['edit'])
        edited_fact = fact.get_fact_lookup_prompt()
        if edited_fact == "The name of the screenwriter of The bomb : the weapon that changed the world is Laurent-Frédéric Bollée.":
            edited_fact = "The name of the screenwriter of  is Laurent-Frédéric Bollée."
        
        
        sub_filter_list = [x for x in filter_list if x[0] == edited_fact]
        # try:
        #     assert len(filtered_list) > 0, edited_fact
        # except AssertionError:
        #     print('Filtered list is empty')
        #     import pdb; pdb.set_trace()
        if len(sub_filter_list) == 0 and len(filter_list) > 0:
            # if this instance is not in the filter list, ignore the instance
            return 
        # for each test query in the example, check if it is in the filter list. If it is not, ignore the test query
        # for tag in ['Relation_Specificity', 'Logical_Generalization', 'Subject_Aliasing', 'Compositionality_I', 'Compositionality_II', 'Forgetfulness']:
        #     tests = d[tag]
            
        #     for test in tests:
        #         import pdb; pdb.set_trace()
                
        making_up_tests = [LookupTestCase.from_dict(test, edited_fact=edited_fact, filter_mode=filter_mode) for test in d['Relation_Specificity']]
        making_up_tests = [x for x in making_up_tests if x is not None]
        
        logical_constraints = [LookupTestCase.from_dict(test, edited_fact=edited_fact, filter_mode=filter_mode) for test in d['Logical_Generalization']]
        logical_constraints = [x for x in logical_constraints if x is not None]
        
        subject_paraphrasing_tests = [LookupTestCase.from_dict(test, edited_fact=edited_fact, filter_mode=filter_mode) for test in d['Subject_Aliasing']]
        subject_paraphrasing_tests = [x for x in subject_paraphrasing_tests if x is not None]
        
        two_hop_tests = [LookupTestCase.from_dict(test, edited_fact=edited_fact, filter_mode=filter_mode) for test in d['Compositionality_I']]
        two_hop_tests = [x for x in two_hop_tests if x is not None]
        
        forward_two_hop_tests = [LookupTestCase.from_dict(test, edited_fact=edited_fact, filter_mode=filter_mode) for test in d['Compositionality_II']]
        forward_two_hop_tests = [x for x in forward_two_hop_tests if x is not None]
        
        prev_storage_tests = [LookupTestCase.from_dict(test, edited_fact=edited_fact, filter_mode=filter_mode) for test in d['Forgetfulness']]
        prev_storage_tests = [x for x in prev_storage_tests if x is not None]
        
        if len(making_up_tests + logical_constraints + subject_paraphrasing_tests + two_hop_tests + forward_two_hop_tests + prev_storage_tests) == 0:
            # if this instance is not in the filter list, ignore the instance
            return
        
        # import pdb; pdb.set_trace()
        
        if d['example_type'] in ['random', 'popular']:
            # import pdb; pdb.set_trace()
            # previous_fact = LookupFact.from_dict(d['edit']['original_fact'])
            return CounterFactualExample(fact, 
                                         None, making_up_tests, logical_constraints,
                                         subject_paraphrasing_tests, two_hop_tests, forward_two_hop_tests, prev_storage_tests)
        elif d['example_type'] == 'recent':
            return RecentlyAddedExample(fact, making_up_tests, logical_constraints, subject_paraphrasing_tests,
                                        two_hop_tests, forward_two_hop_tests, prev_storage_tests)
        else:
            print('Unknown fact type')
            
class CounterFactualExample(Example):

    def __init__(self,
                 fact: Fact,
                 previous_fact: Fact,
                 making_up_tests: list = [],
                 logical_constraints: list = [],
                 subject_paraphrasing_tests: list = [],
                 two_hop_tests: list = [],
                 forward_two_hop_tests: list = [],
                 prev_storage_tests: list = []
                 ):
        super().__init__(
            fact,
            making_up_tests,
            logical_constraints,
            subject_paraphrasing_tests,
            two_hop_tests,
            forward_two_hop_tests,
            prev_storage_tests
        )
        self.previous_fact = previous_fact

    def to_dict(self):
        d = super().create_example_dict('counter_fact')
        d['edit']['original_fact'] = self.previous_fact.to_dict()
        return d

    def __str__(self):
        res = super().__str__()
        res += f'Previous Fact: {str(self.previous_fact)}\n'
        return res


class RecentlyAddedExample(Example):

    def __init__(self,
                 fact: Fact,
                 making_up_tests: list = [],
                 logical_constraints: list = [],
                 subject_paraphrasing_tests: list = [],
                 two_hop_tests: list = [],
                 forward_two_hop_tests: list = [],
                 prev_storage_tests: list = []
                 ):
        super().__init__(
            fact,
            making_up_tests,
            logical_constraints,
            subject_paraphrasing_tests,
            two_hop_tests,
            forward_two_hop_tests,
            prev_storage_tests
        )

    def to_dict(self):
        return super().create_example_dict('recently_added_fact')

def load_jsonlines(fname: str):
    """Read jsonlines file."""
    with open(fname, "r") as f:
        return [json.loads(line) for line in f]

class Dataset:

    def __init__(self, examples: list):
        self.examples = examples

    def sample(self, k: int):
        return random.sample(self.examples, min(k, len(self.examples)))

    def to_file(self, filename):
        p = Path(filename)
        p.parent.mkdir(parents=True, exist_ok=True)
        d = [example.to_dict() for example in self.examples]
        with p.open('w+', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False, indent=2)

    @staticmethod
    def from_file(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        return Dataset([Example.from_dict(example) for example in examples])

    @staticmethod
    def from_jsonl(filename):
        examples = load_jsonlines(filename)
        return Dataset([Example.from_dict(example) for example in examples])
    

class LookupDataset:

    def __init__(self, examples: list, filter_list: list = []):
        self.examples = examples

    def sample(self, k: int):
        return random.sample(self.examples, min(k, len(self.examples)))

    def to_file(self, filename):
        p = Path(filename)
        p.parent.mkdir(parents=True, exist_ok=True)
        d = [example.to_dict() for example in self.examples]
        with p.open('w+', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False, indent=2)

    @staticmethod
    def from_file(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        return Dataset([LookupExample.from_dict(example) for example in examples])

    @staticmethod
    def from_jsonl(filename, filter_list: list = []):
        examples = load_jsonlines(filename)
        dataset = [LookupExample.from_dict(example, filter_list=filter_list) for example in examples[:]]
        dataset = [x for x in dataset if x is not None]
        ret = Dataset(dataset)
        # import pdb; pdb.set_trace()
        return ret
