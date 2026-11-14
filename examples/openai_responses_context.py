"""Explicit application composition; importing this example makes no requests."""
from humansense.text_context_call import TextContextCalls
from humansense.openai_responses import OpenAIResponsesClient
from humansense.responses_evidence import ResponsesEvidenceSession


async def ask_observed_context(human,controller_id,request_id,record,question,*,observer,
                               evidence_directory,api_key,model_revision,account_scope,
                               project_id,maximum_charge):
    """Caller authorizes a cloud call AND separate raw-response/hash retention.

    The already admitted request/observer must name this exact provider/model.
    Example uses native3 task ownership. Native4 uses its bound scope owner.
    Monetary usage verification is deliberately not supplied by this example.
    """
    sink=ResponsesEvidenceSession(evidence_directory,owner_kind='task_native3',
        expected_owner=observer.binding_owner(),retain_response_bytes=True,retain_financial_hashes=True)
    client=OpenAIResponsesClient(api_key=api_key,model_revision=model_revision,
        account_scope=account_scope,project_id=project_id,evidence_session=sink)
    adapter=TextContextCalls(human,controller_id,client,provider='openai',
        model_revision=model_revision,evaluation_observer=observer)
    try:return await adapter.execute(request_id,record,question,maximum_charge=maximum_charge)
    finally:
        try:await adapter.close()
        finally:
            try:await client.close()
            finally:await sink.close()
