"""Controller-selection logic.

Gates choose which controller proposal is executed. Diagnostic gates may use a
rollout-supplied context to resample policies, but they should not step
environments, write replay buffers, or update networks.
"""

from il.gating.action_uncertainty import ActionUncertaintyGate
from il.gating.base import ControllerGate, GateContext
from il.gating.random_gate import RandomGate
from il.gating.expert_q_gap import ExpertQGapGate

__all__ = ["ActionUncertaintyGate", "ControllerGate", "GateContext", "RandomGate", "ExpertQGapGate"]
