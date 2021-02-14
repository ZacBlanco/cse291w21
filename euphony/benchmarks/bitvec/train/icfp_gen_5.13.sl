(set-logic BV)
(synth-fun f ( (x (BitVec 64)) ) (BitVec 64)
((Start (BitVec 64)
((bvnot Start)
(bvxor Start Start)
(bvand Start Start)
(bvor Start Start)
(bvneg Start)
(bvadd Start Start)
(bvmul Start Start)
(bvudiv Start Start)
(bvurem Start Start)
(bvlshr Start Start)
(bvashr Start Start)
(bvshl Start Start)
(bvsdiv Start Start)
(bvsrem Start Start)
(bvsub Start Start)
x
#x0000000000000000
#x0000000000000001
#x0000000000000002
#x0000000000000003
#x0000000000000004
#x0000000000000005
#x0000000000000006
#x0000000000000007
#x0000000000000008
#x0000000000000009
#x0000000000000009
#x0000000000000009
#x000000000000000A
#x000000000000000B
#x000000000000000C
#x000000000000000D
#x000000000000000E
#x000000000000000F
#x0000000000000010
(ite StartBool Start Start)
))
(StartBool Bool
((= Start Start)
(not StartBool)
(and StartBool StartBool)
(or StartBool StartBool)
))))
(constraint (= (f #x82E2CFC24B066408) #x002E2CFC24B06640))
(constraint (= (f #x7153A42A416409B2) #x07153A42A416409B))
(constraint (= (f #x0FD9F387C08C7A62) #x00FD9F387C08C7A6))
(constraint (= (f #x03FBE59CFB1858A2) #x003FBE59CFB1858A))
(constraint (= (f #xA2C725620562317E) #x022C725620562317))
(constraint (= (f #x9FADE97B12C6C046) #x01FADE97B12C6C04))
(constraint (= (f #xA2CE4BC8AC28E31A) #x022CE4BC8AC28E31))
(constraint (= (f #x971C12AA6490E056) #x0171C12AA6490E05))
(constraint (= (f #x229F99A251148A66) #x0229F99A251148A6))
(constraint (= (f #xDC9C7C67A5CA977C) #x05C9C7C67A5CA977))
(constraint (= (f #xEEED10DEC09C00B9) #x07776886F604E005))
(constraint (= (f #xD37E53E32352558F) #x069BF29F191A92AC))
(constraint (= (f #x5EFC0FE6F5BE3413) #x02F7E07F37ADF1A0))
(constraint (= (f #x1257A3C0533E5F67) #x0092BD1E0299F2FB))
(constraint (= (f #x061976DDE0AC0E2B) #x0030CBB6EF056071))
(constraint (= (f #xBB31CB425DAD5D08) #x000044CE34BDA252))
(constraint (= (f #x9640B23DAEE5644A) #x000069BF4DC2511A))
(constraint (= (f #xCBDCB70323A72D12) #x0000342348FCDC58))
(constraint (= (f #x7ADC15B90CC120CA) #x00008523EA46F33E))
(constraint (= (f #xB0EB88613EDD464E) #x00004F14779EC122))
(constraint (= (f #x3F0BDB950BBEA7D1) #x01F85EDCA85DF53E))
(constraint (= (f #x332265EF6D7CB053) #x0199132F7B6BE582))
(constraint (= (f #xB43A5580EBE0FE25) #x05A1D2AC075F07F1))
(constraint (= (f #x5F64C7FE2990EC53) #x02FB263FF14C8762))
(constraint (= (f #x12FD8EE87FFAA1CD) #x0097EC7743FFD50E))
(constraint (= (f #xF50BA2768CCDBF50) #x00000AF45D897332))
(constraint (= (f #x27C012CF4E2FC174) #x0000D83FED30B1D0))
(constraint (= (f #x113DFE31A7ADDF54) #x0000EEC201CE5852))
(constraint (= (f #xED4F5EC8BB43B9C4) #x000012B0A13744BC))
(constraint (= (f #xE73F2B68B793C288) #x000018C0D497486C))
(constraint (= (f #xAAE732B58EFB4847) #x00005518CD4A7104))
(constraint (= (f #xC76F906A9E51785F) #x000038906F9561AE))
(constraint (= (f #x9AB395A8F4116A39) #x0000654C6A570BEE))
(constraint (= (f #xCC1D10C2EB611673) #x000033E2EF3D149E))
(constraint (= (f #xA871D9724A6F2689) #x0000578E268DB590))
(constraint (= (f #xF000000000000001) #x0780000000000000))
(constraint (= (f #xF94C2FD2C635A86B) #x000006B3D02D39CA))
(constraint (= (f #xBBE602AB1F87AB87) #x00004419FD54E078))
(constraint (= (f #xE900CDE66EE2FFFF) #x000016FF3219911D))
(constraint (= (f #x508887F0CB89D6BD) #x0000AF77780F3476))
(constraint (= (f #x24115465D23BF003) #x0000DBEEAB9A2DC4))
(check-synth)
(define-fun f_1 ((x (BitVec 64))) (BitVec 64) (ite (= (bvor #x0000000000000001 x) x) (ite (= (bvashr x x) #x0000000000000000) (ite (= (bvurem x #x0000000000000005) #x0000000000000000) (bvlshr (bvnot x) #x0000000000000010) (ite (= (bvor #x0000000000000009 x) x) (ite (= (bvor #x0000000000000010 x) x) (bvlshr (bvnot x) #x0000000000000010) (bvlshr x #x0000000000000005)) (bvlshr x #x0000000000000005))) (ite (= (bvurem x #x0000000000000003) #x0000000000000000) (bvnot (bvashr x #x0000000000000010)) (ite (= (bvor #x0000000000000003 x) x) (ite (= (bvor #x000000000000000f x) x) (ite (= (bvor #x0000000000000010 x) x) (bvnot (bvashr x #x0000000000000010)) (bvlshr x #x0000000000000005)) (bvnot (bvashr x #x0000000000000010))) (bvlshr x #x0000000000000005)))) (ite (= (bvashr x x) #x0000000000000000) (ite (= (bvand #x000000000000000c x) #x0000000000000000) (bvlshr (bvadd x x) #x0000000000000005) (ite (= (bvor #x0000000000000006 x) x) (bvlshr (bvadd x x) #x0000000000000005) (bvlshr (bvnot x) #x0000000000000010))) (ite (= (bvand #x000000000000000c x) #x0000000000000000) (bvnot (bvashr x #x0000000000000010)) (ite (= (bvor #x0000000000000010 x) x) (bvlshr (bvadd x x) #x0000000000000005) (ite (= (bvurem x #x0000000000000005) #x0000000000000000) (bvnot (bvashr x #x0000000000000010)) (ite (= (bvurem x #x0000000000000006) #x0000000000000000) (bvnot (bvashr x #x0000000000000010)) (ite (= (bvor #x0000000000000006 x) x) (bvlshr (bvadd x x) #x0000000000000005) (ite (= (bvsrem x #x0000000000000003) #x0000000000000000) (bvlshr (bvadd x x) #x0000000000000005) (bvnot (bvashr x #x0000000000000010)))))))))))
