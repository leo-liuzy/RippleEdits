from relation import Relation
from wikidata.utils import get_label
from query import Query, LookupQuery


class Fact:

    def __init__(self, subject_id, relation, target_id):
        self._subject_id = subject_id
        self._relation = relation
        self._target_id = target_id

    def get_subject_label(self):
        return get_label(self._subject_id)

    def get_target_label(self):
        return get_label(self._target_id)

    def get_relation_label(self):
        return self._relation.name.replace('_', ' ')

    def get_fact_query(self):
        return Query(self._subject_id, self._relation, self._target_id)

    def get_fact_prompt(self):
        return self._relation.phrase(get_label(self._subject_id))

    def get_fact_phrased(self):
        return self._relation.phrase(get_label(self._subject_id)) + f' {get_label(self._target_id)}.'

    def to_dict(self):
        return {
            'prompt': self.get_fact_phrased(),
            'subject_id': self._subject_id,
            'relation': self._relation.name,
            'target_id': self._target_id
        }

    @staticmethod
    def from_dict(d):
        return Fact(d['subject_id'], Relation[d['relation']], d['target_id'])

    def __str__(self):
        return f'({self.get_subject_label()}, {self.get_relation_label()}, {self.get_target_label()})'


class LookupFact:
    
    def __init__(self, subject_id, relation, target_id, prompt, subject, target):
        self._subject_id = subject_id
        self._relation = relation
        self._target_id = target_id
        self._prompt = prompt
        self._subject = subject
        self._target = target
        
    def get_subject_label(self):
        return self._subject
        # return get_label(self._subject_id)

    def get_target_label(self):
        return self._target
        # return get_label(self._target_id)

    def get_relation_label(self):
        return self._relation.name.replace('_', ' ')

    def get_fact_prompt(self):
        assert self._target + "." in self._prompt
        
        # prefix (without target)
        prefix = self._prompt.replace(self._target + ".", "").strip()
        return prefix
        
        
    def get_fact_query(self):
        # import pdb; pdb.set_trace()
        assert False, "This should not be called"
        return Query(self._subject_id, self._relation, self._target_id)

    def get_fact_lookup_prompt(self):
        return self._prompt

    def get_fact_phrased(self):
        # import pdb; pdb.set_trace()
        # return self._relation.phrase(get_label(self._subject_id)) + f' {get_label(self._target_id)}.'
        return self._prompt

    def to_dict(self):
        return {
            'prompt': self.get_fact_phrased(),
            'subject_id': self._subject_id,
            'relation': self._relation.name,
            'target_id': self._target_id
        }

    @staticmethod
    def from_dict(d):
        
        try:
            return LookupFact(d['subject_id'], Relation[d['relation']], d['target_id'],
                          prompt=d['prompt'], subject=d["subject"], target=d["target"])
        except KeyError:
            # import pdb; pdb.set_trace()
            # print("HAHAHAHAHA")
            assert False, "This should not be called"
            return Fact(d['subject_id'], Relation[d['relation']], d['target_id'],)
        # return 

    def __str__(self):
        return f'({self.get_subject_label()}, {self.get_relation_label()}, {self.get_target_label()})'
