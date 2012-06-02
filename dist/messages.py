from collections import namedtuple

Task = namedtuple('Task', 'id hash profile anim time')
AddressedTask = namedtuple('AddressedTask', 'addr task')
FullTask = namedtuple('FullTask', 'addr task cubin packer')
