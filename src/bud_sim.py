from enum import Enum


class SimType(Enum):
    USECASE_SIM = "usecase_sim"
    BEST_MODEL_SIM = "best_model_sim"
    BEST_HARDWARE_SIM = "best_hardware_sim"
    PARALLELISATION_STRATEGY_SIM = "parallelisation_strategy_sim"
    HETEROGENEOUS_SIM = "heterogeneous_sim"
    POWER_CONSUMPTION_SIM = "power_consumption_sim"
    COST_SIM = "cost_sim"
    YTD_SIM = "ytd_sim"


class BudSimulator:

    sim_type | SimType | None = None

    def __init__(self, sim_type: SimType = None):
        self.sim_type = sim_type
    
    def set_simulation_type(self, sim_type: SimType):
        self.sim_type = sim_type

    def run(self, **kwargs):
        pass 


    '''------ Simulation Types ------'''

    def get_supported_features(self):
        ''' This must return different supported features like chunked prefill, LoRA, Flash Attention, etc.  and their subfeatures
        that is basically available with the system to accelarate inference on different systems'''




