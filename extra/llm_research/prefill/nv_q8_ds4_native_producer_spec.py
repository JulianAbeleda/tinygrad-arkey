"""Research-only all-native DS4 producer contract."""
from dataclasses import dataclass
@dataclass(frozen=True)
class DS4ProducerSpec:
  M:int=512; K:int=4096; record_values:int=128; record_bytes:int=144; tail_bytes:int=128*144
  threads:int=128
  @property
  def records(self): return self.M*(self.K//self.record_values)
  @property
  def payload_bytes(self): return self.records*self.record_bytes
  @property
  def total_bytes(self): return self.payload_bytes+self.tail_bytes
  @property
  def grid(self): return (self.records,1,1)
  @property
  def block(self): return (self.threads,1,1)
  def q_offset(self,row,k): return ((k//self.record_values)*self.M+row)*self.record_bytes+16+(k%self.record_values)
  def metadata_offset(self,row,k): return ((k//self.record_values)*self.M+row)*self.record_bytes+((k%self.record_values)//32)*4
  def validate(self):
    if self.K%self.record_values or self.record_bytes != 128+16: raise ValueError("DS4 dimensions")
    if (self.K, self.M) not in ((128, 1), (4096, 1), (4096, 512)) and self.M <= 0: raise ValueError("DS4 record geometry")
    if self.records == 16384 and self.payload_bytes != 2359296: raise ValueError("DS4 record geometry")
    if self.q_offset(self.M-1,self.K-1) >= self.payload_bytes: raise ValueError("Q bounds")
    return self
