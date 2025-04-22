from collections import defaultdict

import os
from benchmark import Dataset, Example, TestsAxis, LookupExample, LookupDataset
from modeleditor import ROMEModelEditor, InContextModelEditor, MENDModelEditor, MEMITModelEditor, NoEditModelEditor, KnowledgePropagatorModelEditorLookup, NoEditModelEditorLookup, InContextModelEditorLookup
from queryexecutor import GPT2QueryExecutor, GPT3QueryExecutor, GPTJQueryExecutor, GPTNeoXQueryExecutor, \
    LlamaQueryExecutor, Llama3QueryExecutor, LookupQueryExecutor
from testrunner import ExampleResult
from testrunner import TestRunner, TestResult
from wikidata.utils import write_json
import pdb
from tqdm import tqdm

skip_preconditions = True # We should always skip preconditions for the evaluation

class Evaluator:

    def __init__(self, query_executor, model_editor):
        self._query_executor = query_executor
        self._model_editor = model_editor
        self._test_runner = TestRunner(query_executor, model_editor)

    def average_acc(self, example: Example, test_cases: list, skip_edit: bool = False, skip_restore: bool = False):
        if not len(test_cases) and skip_edit:
            return 0.0, 0.0, 0.0, False
        
        run_res = self._test_runner.run_testcases(example, test_cases, skip_edit=skip_edit, skip_restore=skip_restore, skip_preconditions=skip_preconditions)
        fact_edit_succeeded, res_dict = run_res
        edit_succeeded = True
        if fact_edit_succeeded == ExampleResult.EDIT_FAILED:
            edit_succeeded = False

        if not len(test_cases):
            return 0.0, 0.0, 0.0, edit_succeeded
        # import pdb; pdb.set_trace()
        werent_executed = len(res_dict[TestResult.NOT_EXECUTED])
        successes = len(res_dict[TestResult.PASSED])
        fails = len(res_dict[TestResult.FAILED])
        executed = (successes + fails) / (successes + fails + werent_executed)
        return successes / (successes + fails) if successes else 0.0, executed, len(test_cases), edit_succeeded

    def evaluate_making_up_axis(self, example: Example):
        # relation specificity
        return self.average_acc(example, example.making_up_tests)

    def evaluate_logical_constraints(self, example: Example):
        return self.average_acc(example, example.logical_constraints)

    def evaluate_subject_paraphrasing(self, example: Example):
        # subject aliasing
        return self.average_acc(example, example.subject_paraphrasing_tests)

    def evaluate_two_hop_tests(self, example: Example):
        # Compositionality_I
        return self.average_acc(example, example.two_hop_tests)

    def evaluate_forward_two_hop_tests(self, example: Example):
        # Compositionality_II
        # import pdb; pdb.set_trace()
        return self.average_acc(example, example.forward_two_hop_tests)

    def evaluate_prev_storage_tests(self, example: Example):
        # Forgetfulness
        return self.average_acc(example, example.prev_storage_tests)

    def evaluate(self, example: Example):
        res = defaultdict()
        res[TestsAxis.LOGICAL_CONSTRAINTS] = self.evaluate_logical_constraints(example)
        res[TestsAxis.TWO_HOP] = self.evaluate_two_hop_tests(example)
        res[TestsAxis.FORWARD_TWO_HOP] = self.evaluate_forward_two_hop_tests(example)
        res[TestsAxis.SUBJECT_PARAPHRASING] = self.evaluate_subject_paraphrasing(example)
        res[TestsAxis.MAKING_UP] = self.evaluate_making_up_axis(example)
        res[TestsAxis.PREVIOUS_STORAGE] = self.evaluate_prev_storage_tests(example)
        return res


class ConditionsEvaluator(Evaluator):

    def __init__(self, query_executor):
        super(ConditionsEvaluator, self).__init__(query_executor, None)


if __name__ == '__main__':
    models = [
        'gpt2-medium',
        'gpt2-large',
        'gpt2-xl',
        'gpt-j',
        'gpt-neo',
        'llama', 
        "llama3.1-1b-base-eos-sft"
    ]

    editors = [
        'mend',
        'rome',
        'memit',
        'in-context',
        "no-edit",
        "know-prop",
    ]

    # recently_modified_path = '../data/benchmark/recent.json'
    # fake_facts_path = '../data/benchmark/random.json'
    # top_views_path = '../data/benchmark/popular.json'
    model = 'llama3.1-1b-base-eos-sft-lookup'
    recent_popular_path = f"{os.getenv('PROJ_PLAYGROUND')}/KE-by-CP/data/ripple_edits/meta_train/recent+popular/test.jsonl"
    all_path = f"{os.getenv('PROJ_PLAYGROUND')}/KE-by-CP/data/ripple_edits/meta_train/all/test.jsonl"
    # editor = "know-prop"
    editor = "no-edit"
    # editor = "in-context"
    # get_fact_prompt
    dataset_path = all_path


    if dataset_path == all_path:
        dataset_name = 'all'
    elif dataset_path == recent_popular_path:
        dataset_name = 'recent+popular'
    else:
        raise ValueError(f'Unknown dataset path: {dataset_path}')

    answer_lookup = True
    experiment_name = f'{model}_{editor}_{dataset_name} [answer_lookup={answer_lookup}]'
    print(experiment_name)
    

    # davinvci_query_executor = Llama3QueryExecutor(model_size='text-davinci-003')
    if model == 'llama3.1-1b-base-eos-sft':
        query_executor = Llama3QueryExecutor(model_name_or_path=f"{os.getenv('PROJ_PLAYGROUND')}/mend/models/Llama-3.2-1B-eos-sft", edit_config_name="llama3.2-1B-eos-sft-mid-upper")
    elif model == 'llama3.1-1b-base-eos-sft-lookup':
        query_executor = LookupQueryExecutor(model_name_or_path=f"{os.getenv('PROJ_PLAYGROUND')}/mend/models/Llama-3.2-1B-eos-sft", edit_config_name="llama3.2-1B-eos-sft-mid-upper", use_answer_in_files=answer_lookup)
    else:
        if model == 'llama3.1-1b-base-eos-sft-lookup':
            pass
        else:
            raise ValueError(f'Unknown model: {model}')

    if editor == 'mend':
        model_editor = MENDModelEditor(query_executor)
    elif editor == 'rome':
        model_editor = ROMEModelEditor(query_executor)
    elif editor == 'memit':
        model_editor = MEMITModelEditor(query_executor)
    elif editor == 'in-context':
        # model_editor = InContextModelEditor(query_executor)
        model_editor = InContextModelEditorLookup(query_executor)
    elif editor == "no-edit":
        # model_editor = NoEditModelEditor(query_executor)
        model_editor = NoEditModelEditorLookup(query_executor)
    elif editor == "know-prop":
        model_editor = KnowledgePropagatorModelEditorLookup(query_executor)
    else:
        raise ValueError(f'Unknown model editor: {editor}')
    # import pdb; pdb.set_trace()
    
    evaluator = Evaluator(query_executor=query_executor, model_editor=model_editor)
    dataset = LookupDataset.from_jsonl(dataset_path)

    precisions_json = dict()
    # num_of_examples = 200
    examples_for_eval = dataset.examples[:]
    # examples_for_eval = dataset.sample(num_of_examples)
    eval_size = len(examples_for_eval)

    succeeded_edits = defaultdict(lambda: 0)
    average_precision = defaultdict(lambda: 0)
    average_executed = defaultdict(lambda: 0)
    average_size = defaultdict(lambda: 0)
    total_checked_examples = defaultdict(lambda: 0)
    executed_portion_dict = defaultdict(lambda: 0)
    # import pdb; pdb.set_trace()
    for i, example in tqdm(enumerate(examples_for_eval), total=eval_size):
        if (i + 1) % 10 == 0:
            print(f'{i + 1}/{eval_size}')

        # if example.fact.get_subject_label() == '' or example.fact.get_target_label() == '':
        #     print(f'Skipping example: {example.to_dict()}')
        #     continue

        evaluation_results = evaluator.evaluate(example)

        res_dict_for_json = dict()
        for propagation_type, results in evaluation_results.items():
            precision, executed, size, edit_succeeded = results
            # if executed == 0.0:
                # continue
            # if edit_succeeded:
            # succeeded_edits[axis] += 1
            average_precision[propagation_type] += precision
            res_dict_for_json[propagation_type.name] = precision
            average_executed[propagation_type] += executed
            average_size[propagation_type] += size
            # precisions_json[str(example.fact)] = precision
            total_checked_examples[propagation_type] += 1

        precisions_json[str(example.fact)] = res_dict_for_json

        for propagation_type in TestsAxis:
            if propagation_type in evaluation_results:
                executed_portion_dict[propagation_type] += evaluation_results[propagation_type][1]
    
    print(experiment_name)
    res_str = f'skip_preconditions={skip_preconditions}\n'
    print(res_str)
    for propagation_type in TestsAxis:
        print(f'Results of axis {propagation_type}:')
        res_str += f'Results of axis {propagation_type}:\n'

        if total_checked_examples[propagation_type] == 0:
            print(f'No checked tests for this axis')
            res_str += f'No checked tests for this axis\n'
            continue
        
        # pdb.set_trace()
        # if succeeded_edits[axis] > 0:
        #     average_precision[axis] /= succeeded_edits[axis]
        #     average_executed[axis] /= succeeded_edits[axis]
        #     average_size[axis] /= succeeded_edits[axis]
            
        assert total_checked_examples[propagation_type] > 0
        average_precision[propagation_type] /= total_checked_examples[propagation_type]
        average_executed[propagation_type] /= total_checked_examples[propagation_type]
        average_size[propagation_type] /= total_checked_examples[propagation_type]
            

        # print(f'{succeeded_edits[axis]} successful edits (out of {eval_size})')
        # res_str += f'{succeeded_edits[axis]} successful edits (out of {eval_size})\n'
        print(f'Average accuracy is {average_precision[propagation_type]}')
        res_str += f'Average accuracy is {average_precision[propagation_type]}\n'
        # print(f'Average portion of executed_tests is {average_executed[propagation_type]}')
        # res_str += f'Average portion of executed_tests is {average_executed[propagation_type]}\n'
        print(f'Average total number of tests is {average_size[propagation_type]}')
        res_str += f'Average total number of tests is {average_size[propagation_type]}\n'
        print("total_checked_examples[propagation_type]: ", total_checked_examples[propagation_type], "\n")
        res_str += f'total_checked_examples[propagation_type]: {total_checked_examples[propagation_type]}\n'

    write_json(precisions_json, f'./{experiment_name}_res_2.json')

    with open(f'./{experiment_name}_2.txt', 'w+', encoding='utf-8') as f:
        f.write(res_str)
