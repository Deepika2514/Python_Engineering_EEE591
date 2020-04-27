Inverter Circuit - this is still a comment
.options list node post
.tran 200p 20n
.print tran v(in) v(out)
m1 out in vcc vcc pch l=1u w=20u
m2 out in 0 0 nch l=1u w=20u
vcc vcc 0 5
vin in 0 0 pulse .2 4.8 2n 1n 1n 5n 20n
cload out 0 .75p
.model pch pmos level=1
.model nch nmos level=1
.end
