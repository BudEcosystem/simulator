from enum import Enum
from pydantic import BaseModel, Field
from GenZ.simulation import SimulationEngine, SimulationConfig

class SimType(Enum):
    USECASE_SIM = "usecase_sim"
    BEST_MODEL_SIM = "best_model_sim"
    BEST_HARDWARE_SIM = "best_hardware_sim"
    PARALLELISATION_STRATEGY_SIM = "parallelisation_strategy_sim"
    HETEROGENEOUS_SIM = "heterogeneous_sim"
    POWER_CONSUMPTION_SIM = "power_consumption_sim"
    COST_SIM = "cost_sim"
    YTD_SIM = "ytd_sim"


class SimulationConfig(BaseModel):
    models: list[dict] = Field(description="The model id to simulate", default=None)
    batch_size: int = Field(description="The batch size to simulate", default=None)
    precision: str = Field(description="The precision to simulate", default=None)
    decode_length: int = Field(description="The decode length to simulate", default=None)
    usecases: list[dict] = Field(description="The usecases to simulate", default=None)
    hardwares: list[dict] = Field(description="The hardwares to simulate", default=None)
    features: list[dict] = Field(description="The features to simulate", default=[])



class BudSimulator:

    sim_type | SimType | None = None

    def __init__(self, sim_type: SimType = None):
        self.sim_type = sim_type
    
    def set_simulation_type(self, sim_type: SimType):
        self.sim_type = sim_type

    def run(self, **kwargs):
        pass 

    def get_supported_features(self):
        ''' This must return different supported features like chunked prefill, LoRA, Flash Attention, etc.  and their subfeatures
        that is basically available with the system to accelarate inference on different systems'''
        


    '''------ Simulation Types ------'''

    def get_supported_features(self):
        ''' This must return different supported features like chunked prefill, LoRA, Flash Attention, etc.  and their subfeatures
        that is basically available with the system to accelarate inference on different systems'''

    
    def execute_standard_sim(self, **kwargs):




