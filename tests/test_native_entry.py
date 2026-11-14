import unittest
from dataclasses import FrozenInstanceError
from humansense.native_entry import _issue_entry_binding,NativeEntryBinding


def owner():
    return dict(comparison_plan_sha256='1'*64,collection_sha256='2'*64,stream_protocol_sha256='3'*64,
        reconciliation_protocol_sha256='4'*64,instrumentation_protocol_sha256=None,scope_kind='task',task_id='fixture-task',
        arm='C',stream_id='fixture-stream',input_sha256='5'*64,recording_sha256='6'*64,configuration_sha256='7'*64,cost_sha256='8'*64)


def binding(prompt='9'*64):
    return dict(schema_version=1,owner_kind='task_native3',owner=owner(),entry=dict(operation_id='fixture-op',attempt_id='fixture-attempt',
        execution_instance_sha256='a'*64,execution_sha256='b'*64),provider='openai',model_revision='fixture-model',role='text_llm',prompt_sha256=prompt)


class Entry(unittest.TestCase):
    def test_issue_owns_immutable_snapshot(self):
        row=binding();value=_issue_entry_binding(row);row['owner']['arm']='A'
        view=value.snapshot();view['owner']['arm']='B'
        self.assertEqual(value.snapshot()['owner']['arm'],'C')
        with self.assertRaises(FrozenInstanceError):value._bytes=b'{}'
        with self.assertRaises(TypeError):NativeEntryBinding(binding())
    def test_tagged_union_rejects_fabricated_schema(self):
        row=binding();row['entry']['entry_ordinal']=0
        with self.assertRaises(ValueError):_issue_entry_binding(row)
        row=binding();row['owner_kind']='recording_native4'
        with self.assertRaises(ValueError):_issue_entry_binding(row)
    def test_recording_binding_preserves_actual_entry(self):
        row=binding();row['owner_kind']='recording_native4'
        row['owner']=dict(context=dict(schema_version=4,kind='recording_native',nonce='c'*64,invocation_ordinal=0,
            collection_sha256='d'*64,scope_protocol_sha256='e'*64,instrumentation_protocol_sha256='f'*64),scope_index=1,binding_index=0)
        row['entry']=dict(ticket_ordinal=2,entry_ordinal=1,instance_sha256='1'*64,request_decision_sha256='2'*64,entered_event_sha256='3'*64)
        self.assertEqual(_issue_entry_binding(row).snapshot(),row)
