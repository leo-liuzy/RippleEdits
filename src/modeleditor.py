import sys
import os
import torch
import pandas as pd
from queryexecutor import QueryExecutor
from copy import deepcopy


class ModelEditor:

    def __init__(self, query_executor):
        self._query_executor = query_executor
        self._model = self._query_executor.get_model()
        self._tokenizer = self._query_executor.get_tokenizer()
        self._model_name = self._query_executor.get_model_name()
        self._model_device = self._query_executor.get_device()

    def edit_model(self, fact):
        raise NotImplementedError()  # Override in concrete classes

    def restore_model(self):
        raise NotImplementedError()  # Override in concrete classes


class InContextModelEditor(ModelEditor):

    def __init__(self, query_executor: QueryExecutor):
        super().__init__(query_executor)

    def edit_model(self, fact):
        edit_fact = fact.get_fact_phrased()
        
        context = 'Imagine that ' + edit_fact[0].lower() + edit_fact[1:] + ' '
        print(f'In Context Editing added context: {context}')
        self._query_executor.set_prompt_context(context)

    def restore_model(self):
        self._query_executor.set_prompt_context('')
        

class NoEditModelEditor(ModelEditor):

    def __init__(self, query_executor: QueryExecutor):
        super().__init__(query_executor)

    def edit_model(self, fact):
        context = ""
        self._query_executor.set_prompt_context(context)

    def restore_model(self):
        self._query_executor.set_prompt_context('')


class RomeStyleModelEditor(ModelEditor):

    def __init__(self, query_executor):
        self._changed_weights = None
        super().__init__(query_executor)

    @staticmethod
    def _format_fact_for_rome(fact):
        subject = fact.get_subject_label()
        target = fact.get_target_label()
        prompt = fact.get_fact_prompt().replace(subject, '{}')
        return [{'prompt': prompt, 'subject': subject, 'target_new': {'str': target}}]

    def edit_model(self, fact):
        raise NotImplementedError()  # Override in concrete classes

    def restore_model(self):
        if self._changed_weights is None:
            return

        sys.path.append(f"{os.getenv('PROJ_PLAYGROUND')}/memit_original")
        # import pdb; pdb.set_trace()
        from util import nethook

        with torch.no_grad():
            for k, v in self._changed_weights.items():
                nethook.get_parameter(self._model, k)[...] = v.to(self._model_device)

        # sys.path.remove('..')
        # os.chdir('../..')


class MEMITModelEditor(RomeStyleModelEditor):

    def __init__(self, query_executor):
        super().__init__(query_executor)

    def edit_model(self, fact):
        
        # os.chdir('/u/zliu/datastor1/RippleEdits/src/memit')
        sys.path.append(f"{os.getenv('PROJ_PLAYGROUND')}/memit_original")
        from memit import MEMITHyperParams, apply_memit_to_model
        requests = self._format_fact_for_rome(fact)
        hparams = MEMITHyperParams.from_json(f"{os.getenv('PROJ_PLAYGROUND')}/memit_original/hparams/MEMIT/{self._model_name}.json")
        _, self._changed_weights = apply_memit_to_model(self._model, self._tokenizer, requests, hparams, return_orig_weights=True)


class ROMEModelEditor(RomeStyleModelEditor):

    def __init__(self, query_executor):
        super().__init__(query_executor)

    def edit_model(self, fact):
        os.chdir('./memit')
        sys.path.append('..')
        from rome import ROMEHyperParams, apply_rome_to_model

        requests = self._format_fact_for_rome(fact)
        hparams = ROMEHyperParams.from_json(f'hparams/ROME/{self._model_name}.json')
        _, self._changed_weights = apply_rome_to_model(self._model, self._tokenizer, requests, hparams, return_orig_weights=True)

        sys.path.remove('..')
        os.chdir('../..')


class MENDModelEditor(RomeStyleModelEditor):

    def __init__(self, query_executor):
        super().__init__(query_executor)

    def edit_model(self, fact):
        os.chdir('./memit')
        sys.path.append('..')
        from baselines.mend import MENDHyperParams, MendRewriteExecutor

        requests = self._format_fact_for_rome(fact)
        hparams = MENDHyperParams.from_json(f'hparams/MEND/{self._model_name}.json')
        _, self._changed_weights = MendRewriteExecutor().apply_to_model(self._model, self._tokenizer, requests, hparams, return_orig_weights=True)

        sys.path.remove('..')
        os.chdir('../..')

class EditorLookup(RomeStyleModelEditor):
    
    def __init__(self, query_executor, ice=False):
        super().__init__(query_executor)
        self._ice = ice

    def edit_model(self, fact):
        # from baselines.mend import MENDHyperParams, MendRewriteExecutor
        assert hasattr(self, "_column_name")
        edit_fact = fact.get_fact_lookup_prompt()
        if edit_fact == "The name of the screenwriter of The bomb : the weapon that changed the world is Laurent-Frédéric Bollée.":
            edit_fact = "The name of the screenwriter of  is Laurent-Frédéric Bollée."
        edit_subdf_content = []
        # import pdb; pdb.set_trace()
        for i, r in self.df.iterrows():
            if edit_fact in r[self._column_name]:
                edit_subdf_content.append(r)
        edit_subdf = pd.DataFrame(edit_subdf_content)
        # assert len(edit_subdf) > 0
        if len(edit_subdf) == 0:
            import pdb; pdb.set_trace()
       
        assert len(edit_subdf["id"].unique()) == 1
        question2generated_answer = {}
        questions = edit_subdf["question"].tolist()
        if self._ice:
            tmp = "Imagine that " + edit_fact[0].lower() + edit_fact[1:]
            assert all([tmp in q for q in questions])
            # import pdb; pdb.set_trace()
            questions = [q.replace(tmp, "", 1).strip() for q in questions]
        
        predicted_answers = edit_subdf["predicted_answer"].tolist()
        for q_i, q in enumerate(questions):
            if q not in question2generated_answer:
                question2generated_answer[q] = str(predicted_answers[q_i])
            else:
                assert question2generated_answer[q] == str(predicted_answers[q_i])
                
        self._query_executor._lookup_table = deepcopy(question2generated_answer)
        # import pdb; pdb.set_trace()
        
    
class NoEditModelEditorLookup(EditorLookup):

    def __init__(self, query_executor, ice=False):
        super().__init__(query_executor, ice)
        self.df = pd.read_excel(f"{os.getenv('PROJ_PLAYGROUND')}/mend/ripple_exp_output/llama3.2-1B-eos-sft/all/base_n=500_prompt=no_w-gen_wo-icl_ice=False.xlsx")
        self._column_name = "input"

class InContextModelEditorLookup(EditorLookup):
    
    def __init__(self, query_executor, ice=True):
        super().__init__(query_executor, ice)
        self.df = pd.read_excel(f"{os.getenv('PROJ_PLAYGROUND')}/mend/ripple_exp_output/llama3.2-1B-eos-sft/all/base_n=500_prompt=no_w-gen_wo-icl_ice=True.xlsx")
        self._column_name = "input"

class KnowledgePropagatorModelEditorLookup(EditorLookup):

    def __init__(self, query_executor, experiment_name="ripple_edits_all_heavy-noshare-mid-upper3_all-in-outer", ice=False):
        super().__init__(query_executor, ice)
        self.df = pd.read_excel(f"{os.getenv('PROJ_PLAYGROUND')}/mend/ripple_exp_output/{experiment_name}/ripple_edits/mend_eval_loss=clm_input=seen_n=500_prompt=no_w-gen_wo-icl_e+s_all-question.xlsx")
        self._column_name = "edit_input"
    
    
class CPTModelEditorLookup(EditorLookup):

    def __init__(self, query_executor, experiment_name="Llama-3.2-1B-eos-sft_clm-baseline_lr=1e-05_epoch=4.0_tuned-params=midupper3-mlp", ice=False):
        super().__init__(query_executor, ice)
        self.df = pd.read_excel(f"{os.getenv('PROJ_PLAYGROUND')}/mend/ripple_exp_output/{experiment_name}/all_results_ALL_with_input.xlsx")
        self._column_name = "edit_input"


class MENDModelEditorLookup(EditorLookup):

    def __init__(self, query_executor, experiment_name="zereFull_original_mend_share_top3", ice=False):
        super().__init__(query_executor, ice)
        self.df = pd.read_excel(f"{os.getenv('PROJ_PLAYGROUND')}/mend/ripple_exp_output/{experiment_name}/ripple_edits/mend_eval_loss=sft_input=seen_n=500_prompt=no_w-gen_wo-icl_e+s_all-question.xlsx")
        self._column_name = "edit_input"
        
    def edit_model(self, fact):
        # from baselines.mend import MENDHyperParams, MendRewriteExecutor
        assert hasattr(self, "_column_name")
        edit_fact = fact.get_fact_lookup_prompt()
        
        if edit_fact == "The name of the screenwriter of  is Laurent-Frédéric Bollée.":
            edit_fact = "The name of the screenwriter of The bomb : the weapon that changed the world is Laurent-Frédéric Bollée."

        
        if edit_fact.endswith("."):
            edit_fact = edit_fact[:-1]
        edit_subdf_content = []
        # import pdb; pdb.set_trace()
        for i, r in self.df.iterrows():
            if edit_fact in r[self._column_name]:
                edit_subdf_content.append(r)
        edit_subdf = pd.DataFrame(edit_subdf_content)
        # if len(edit_subdf) == 0:
            # import pdb; pdb.set_trace()
        assert len(edit_subdf) > 0

        assert len(edit_subdf["id"].unique()) == 1
        question2generated_answer = {}
        questions = edit_subdf["question"].tolist()
        if self._ice:
            tmp = "Imagine that " + edit_fact[0].lower() + edit_fact[1:]
            assert all([tmp in q for q in questions])
            # import pdb; pdb.set_trace()
            questions = [q.replace(tmp, "", 1).strip() for q in questions]
        
        predicted_answers = edit_subdf["predicted_answer"].tolist()
        for q_i, q in enumerate(questions):
            if q not in question2generated_answer:
                question2generated_answer[q] = str(predicted_answers[q_i])
            else:
                assert question2generated_answer[q] == str(predicted_answers[q_i])
                
        self._query_executor._lookup_table = deepcopy(question2generated_answer)
        # import pdb; pdb.set_trace()
        
class MEMITModelEditorLookup(EditorLookup):
    def __init__(self, query_executor, experiment_name="llama3.2-1B-eos-sft-mid-upper", ice=False):
        super().__init__(query_executor, ice)
        self.df = pd.read_excel(f"{os.getenv('PROJ_PLAYGROUND')}/mend/ripple_exp_output/{experiment_name}/ripple_edits/memit(ripple_all)_eval_loss=clm_input=seen_n=500_prompt=no_w-gen_wo-icl_e+s_all-question.xlsx")
        self.base_df = pd.read_excel(f"{os.getenv('PROJ_PLAYGROUND')}/mend/ripple_exp_output/llama3.2-1B-eos-sft/all/base_n=500_prompt=no_w-gen_wo-icl_ice=False.xlsx")
        
        self._column_name = "edit_input"
        self._base_column_name = "input"
        
    def edit_model(self, fact):
        # from baselines.mend import MENDHyperParams, MendRewriteExecutor
        assert hasattr(self, "_column_name")
        edit_fact = fact.get_fact_lookup_prompt()
        
        if edit_fact == "The name of the screenwriter of  is Laurent-Frédéric Bollée.":
            edit_fact = "The name of the screenwriter of The bomb : the weapon that changed the world is Laurent-Frédéric Bollée."

        
        edit_subdf_content = []
        # import pdb; pdb.set_trace()
        for i, r in self.df.iterrows():
            if edit_fact in r[self._column_name]:
                edit_subdf_content.append(r)
        edit_subdf = pd.DataFrame(edit_subdf_content)
        if len(edit_subdf) == 0:
            # import pdb; pdb.set_trace()
            edit_subdf_content = []
            # import pdb; pdb.set_trace()
            for i, r in self.base_df.iterrows():
                if edit_fact in r[self._base_column_name]:
                    edit_subdf_content.append(r)
            edit_subdf = pd.DataFrame(edit_subdf_content)
        assert len(edit_subdf) > 0

        assert len(edit_subdf["id"].unique()) == 1
        question2generated_answer = {}
        questions = edit_subdf["question"].tolist()
        
        predicted_answers = edit_subdf["predicted_answer"].tolist()
        for q_i, q in enumerate(questions):
            if q not in question2generated_answer:
                question2generated_answer[q] = str(predicted_answers[q_i])
            else:
                assert question2generated_answer[q] == str(predicted_answers[q_i])
                
        self._query_executor._lookup_table = deepcopy(question2generated_answer)