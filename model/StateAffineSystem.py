from model.BaseModel import ReservoirModel
from model.reservoirs import SASReservoir, TrigoSASReservoir


class StateAffineSystem(ReservoirModel):

    _reservoir_class = SASReservoir


class TrigoStateAffineSystem(ReservoirModel):

    _reservoir_class = TrigoSASReservoir
