# Scaling ladders: one parameterized family per conic class, three rungs.
#
# Purpose is per-phase scaling evidence on a development machine -- each rung
# grows the dimension that stresses the class's core linear algebra (LP/SOC
# normal equations, SDP Schur complement) so a regression shows up as a phase
# ratio rather than only a total time. These rows make no production-scale
# claim; cluster-scale evidence stays with the large suite.
#
# Reference objectives are empirically verified (Mac, Float64/auto):
# lp_box optimum is sum(1:N); soc_dimension optimum is sqrt(N-1) (least-norm
# equality solution); sdp_many_small optimum is the variable count.

const LADDER_SPECS = BenchmarkSpec[
    _synthetic(
        "ladder/lp_100", :lp, :linear_program, (:ladder,),
        (:scaling, :ladder), :lp_scaling_ladder_small, 211,
        :lp_box, (variables=100,); objective=5050.0,
        size=(variables=100, inequalities=200, equalities=0),
    ),
    _synthetic(
        "ladder/lp_300", :lp, :linear_program, (:ladder,),
        (:scaling, :ladder), :lp_scaling_ladder_medium, 223,
        :lp_box, (variables=300,); objective=45150.0,
        size=(variables=300, inequalities=600, equalities=0),
    ),
    _synthetic(
        "ladder/lp_1000", :lp, :linear_program, (:ladder,),
        (:scaling, :ladder), :lp_scaling_ladder_large, 227,
        :lp_box, (variables=1000,); objective=500500.0,
        size=(variables=1000, inequalities=2000, equalities=0),
    ),
    _synthetic(
        "ladder/soc_128", :socp, :second_order_cone_program, (:ladder,),
        (:scaling, :ladder), :soc_scaling_ladder_small, 229,
        :soc_dimension, (dimension=128,); objective=sqrt(127.0),
        size=(variables=128, cone_dimensions=(128,), equalities=127),
    ),
    _synthetic(
        "ladder/soc_512", :socp, :second_order_cone_program, (:ladder,),
        (:scaling, :ladder), :soc_scaling_ladder_medium, 233,
        :soc_dimension, (dimension=512,); objective=sqrt(511.0),
        size=(variables=512, cone_dimensions=(512,), equalities=511),
    ),
    _synthetic(
        "ladder/soc_2048", :socp, :second_order_cone_program, (:ladder,),
        (:scaling, :ladder), :soc_scaling_ladder_large, 239,
        :soc_dimension, (dimension=2048,); objective=sqrt(2047.0),
        size=(variables=2048, cone_dimensions=(2048,), equalities=2047),
    ),
    _synthetic(
        "ladder/sdp_250", :sdp, :semidefinite_program, (:ladder,),
        (:scaling, :ladder), :sdp_scaling_ladder_small, 241,
        :sdp_many_small, (blocks=250,); objective=250.0,
        size=(variables=250, blocks=250, block_dimension=2),
    ),
    _synthetic(
        "ladder/sdp_600", :sdp, :semidefinite_program, (:ladder,),
        (:scaling, :ladder), :sdp_scaling_ladder_medium, 251,
        :sdp_many_small, (blocks=600,); objective=600.0,
        size=(variables=600, blocks=600, block_dimension=2),
    ),
    _synthetic(
        "ladder/sdp_1000", :sdp, :semidefinite_program, (:ladder,),
        (:scaling, :ladder), :sdp_scaling_ladder_large, 257,
        :sdp_many_small, (blocks=1000,); objective=1000.0,
        size=(variables=1000, blocks=1000, block_dimension=2),
    ),
]
