from .BaseModel import ReservoirModel
from .reservoirs import SASReservoir, TrigoSASReservoir


class StateAffineSystem(ReservoirModel):

    _reservoir_class = SASReservoir

    def __init__(self, *args, W_in=None, **kwargs):
        super().__init__(*args, W_in=1, **kwargs)


class TrigoStateAffineSystem(StateAffineSystem):

    _reservoir_class = TrigoSASReservoir
