"""Application composition: explicit recorded BGR capture and existing async clients.

No default network endpoint, model or camera is activated. Client dictionaries
contain completion/provider/model_revision. The application verifies ceilings and
receipts and retains its CallBudget/CallController; this helper never resets it.
"""
from humansense.fill_conflict import FillConflictRuntime


async def inspect_recorded(human,controller_id,frame,*,fill_config,small_client,strong_client=None,
                           strong_opt_in=False,charge_ceilings,chain_ceiling,unit,pricing_revision,
                           evaluation_observer=None):
    driver=FillConflictRuntime(human,'recorded_fill',controller_id,fill_config=fill_config,
        small_client=small_client,strong_client=strong_client,strong_opt_in=strong_opt_in,
        charge_ceilings=charge_ceilings,chain_ceiling=chain_ceiling,unit=unit,pricing_revision=pricing_revision,
        evaluation_observer=evaluation_observer)
    try:
        return await driver.inspect(frame)
    finally:
        await driver.close()
