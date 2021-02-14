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
(constraint (= (f #x56ac845de994e72c) #x56ac845de994e72d))
(constraint (= (f #xd053ad358d43baa0) #xd053ad358d43baa1))
(constraint (= (f #x4cb34b0591cb895c) #x4cb34b0591cb895d))
(constraint (= (f #x3d71139ed39e874a) #x3d71139ed39e874b))
(constraint (= (f #xa1821d09b93b50bb) #xbcfbc5ec8d895e89))
(constraint (= (f #x6e57889aea46ce86) #x6e57889aea46ce87))
(constraint (= (f #x4998b16e5beb480a) #x6cce9d2348296feb))
(constraint (= (f #x9496e8d0be525139) #xd6d22e5e835b5d8d))
(constraint (= (f #x9e9b71e29acaab15) #x9e9b71e29acaab16))
(constraint (= (f #x9cced78c3d34ee98) #x9cced78c3d34ee99))
(constraint (= (f #x47db5dd1e5be131b) #x7049445c3483d9c9))
(constraint (= (f #x32eabe424a662b22) #x9a2a837b6b33a9bb))
(constraint (= (f #xea866ebd042c83d8) #xea866ebd042c83d9))
(constraint (= (f #xe4deeb2736d22dae) #x364229b1925ba4a3))
(constraint (= (f #x7eedd487be1eae7a) #x7eedd487be1eae7b))
(constraint (= (f #xe17b7910d9ae0dc0) #x3d090dde4ca3e47f))
(constraint (= (f #x6de599de1774cc33) #x6de599de1774cc34))
(constraint (= (f #xe511dd9d7ecd8ec3) #xe511dd9d7ecd8ec4))
(constraint (= (f #x28bab9e73d907666) #xae8a8c3184df1333))
(constraint (= (f #x22e0bad7d59be7bb) #x22e0bad7d59be7bc))
(constraint (= (f #xadde95c9b426b26d) #xadde95c9b426b26e))
(constraint (= (f #xd1e3863ab75eb240) #xd1e3863ab75eb241))
(constraint (= (f #x19e7a2ee954b9be9) #x19e7a2ee954b9bea))
(constraint (= (f #x8706c5eebe4c879c) #x8706c5eebe4c879d))
(constraint (= (f #xc38acd6aec9e40c7) #x78ea652a26c37e71))
(constraint (= (f #x5cead0940432061b) #x462a5ed7f79bf3c9))
(constraint (= (f #x0d72c9e7ee16e180) #x0d72c9e7ee16e181))
(constraint (= (f #x914ee8be8824204e) #xdd622e82efb7bf63))
(constraint (= (f #x47b9b755eed5388a) #x708c915422558eeb))
(constraint (= (f #xa1ac41a402dad951) #xa1ac41a402dad952))
(constraint (= (f #xe021added3b81d29) #x3fbca442588fc5ad))
(constraint (= (f #xd23808bc3db970b4) #x5b8fee87848d1e97))
(constraint (= (f #x1e686622254ebd10) #x1e686622254ebd11))
(constraint (= (f #x165214b3e2bdeee9) #x165214b3e2bdeeea))
(constraint (= (f #x5c2ea6b6e7e4ca5c) #x5c2ea6b6e7e4ca5d))
(constraint (= (f #xde0bc44962c426e8) #x43e8776d3a77b22f))
(constraint (= (f #x7ea74b5ea7e0b48b) #x7ea74b5ea7e0b48c))
(constraint (= (f #x6baebde3d3bcd86e) #x6baebde3d3bcd86f))
(constraint (= (f #x0dbd81a0291e9e46) #x0dbd81a0291e9e47))
(constraint (= (f #x99a761493e1b71b6) #xccb13d6d83c91c93))
(constraint (= (f #x3c74e170619ee128) #x3c74e170619ee129))
(constraint (= (f #x062cec2e6dbece8e) #x062cec2e6dbece8f))
(constraint (= (f #x9a74abe4ee216ba8) #xcb16a83623bd28af))
(constraint (= (f #x3a09e9d32b220e5e) #x8bec2c59a9bbe343))
(constraint (= (f #x1b6ac6b8e04356e7) #xc92a728e3f795231))
(constraint (= (f #x957ee9ece3e8e75e) #x957ee9ece3e8e75f))
(constraint (= (f #x5c7e53411119584c) #x4703597dddcd4f67))
(constraint (= (f #x220284d49262e804) #x220284d49262e805))
(constraint (= (f #x85c26cace04adc8a) #x85c26cace04adc8b))
(constraint (= (f #xe9d33aac96e188c2) #xe9d33aac96e188c3))
(constraint (= (f #x3704eee2e2b150d1) #x91f6223a3a9d5e5d))
(constraint (= (f #xde22118a4ec8a2a7) #xde22118a4ec8a2a8))
(constraint (= (f #x1ee642dd297cea75) #x1ee642dd297cea76))
(constraint (= (f #xb2cd4b9d0ddc2ae6) #x9a6568c5e447aa33))
(constraint (= (f #x74ee55bb68a92d55) #x162354892eada555))
(constraint (= (f #x8e01a83a24217ac9) #xe3fcaf8bb7bd0a6d))
(constraint (= (f #xa62362de9d6bde06) #xa62362de9d6bde07))
(constraint (= (f #xd38a5ed37e53de17) #xd38a5ed37e53de18))
(constraint (= (f #x7cc781eaa5572e65) #x0670fc2ab551a335))
(constraint (= (f #x61408cb7e6a4e908) #x61408cb7e6a4e909))
(constraint (= (f #xbc7dd5e1d19a46b7) #x8704543c5ccb7291))
(constraint (= (f #x5a42dee5c1ecdd88) #x5a42dee5c1ecdd89))
(constraint (= (f #xba39358ceaad3abc) #x8b8d94e62aa58a87))
(constraint (= (f #x6266403a3bbbe6ab) #x6266403a3bbbe6ac))
(constraint (= (f #x1e1b247e00495053) #xc3c9b703ff6d5f59))
(constraint (= (f #xaea77d69657825a9) #xa2b1052d350fb4ad))
(constraint (= (f #xcacdaca74e911675) #x6a64a6b162ddd315))
(constraint (= (f #xed21823725ee6093) #x25bcfb91b4233ed9))
(constraint (= (f #xc478990a8c04d59e) #xc478990a8c04d59f))
(constraint (= (f #x33c97c8024ed7e78) #x986d06ffb625030f))
(constraint (= (f #xeb1e5d4991c22a8e) #x29c3456cdc7baae3))
(constraint (= (f #x5700924658e7b698) #x5700924658e7b699))
(constraint (= (f #x1b9960c7849c90e4) #x1b9960c7849c90e5))
(constraint (= (f #xe1891b4e60345e0b) #x3cedc9633f9743e9))
(constraint (= (f #xdee4752ee9de0c09) #x423715a22c43e7ed))
(constraint (= (f #x4285ce35988a7451) #x7af46394ceeb175d))
(constraint (= (f #xa31ee354eb41a3a4) #xa31ee354eb41a3a5))
(constraint (= (f #x696e43d7ce8b0134) #x2d23785062e9fd97))
(constraint (= (f #x4b5cd3aeed61e57a) #x4b5cd3aeed61e57b))
(constraint (= (f #xe3a97b1de20e8a35) #xe3a97b1de20e8a36))
(constraint (= (f #xa2cced5ed37eeb2e) #xa2cced5ed37eeb2f))
(constraint (= (f #x6e7e9d20952772d7) #x2302c5bed5b11a51))
(constraint (= (f #x2ed299bede64b80c) #x2ed299bede64b80d))
(constraint (= (f #x2aee9b370cebbe00) #x2aee9b370cebbe01))
(constraint (= (f #x9b2188d24d003b23) #xc9bcee5b65ff89b9))
(constraint (= (f #xde1eab839300b57e) #xde1eab839300b57f))
(constraint (= (f #x8890ee9e8c43d49c) #x8890ee9e8c43d49d))
(constraint (= (f #xa1c89eaee1a57146) #xbc6ec2a23cb51d73))
(constraint (= (f #xe29e77eeeb498e0e) #xe29e77eeeb498e0f))
(constraint (= (f #x3486ecccd237cbec) #x3486ecccd237cbed))
(constraint (= (f #x8ea984ac2ab1b8d5) #x8ea984ac2ab1b8d6))
(constraint (= (f #x21ca84c10794de34) #x21ca84c10794de35))
(constraint (= (f #xed78e35276a4dee6) #xed78e35276a4dee7))
(constraint (= (f #x54b2585e2e0c153b) #x569b4f43a3e7d589))
(constraint (= (f #x39b049d52bee10ce) #x8c9f6c55a823de63))
(constraint (= (f #x778e9e7118ae0131) #x10e2c31dcea3fd9d))
(constraint (= (f #xe5e4836975be778b) #x3436f92d148310e9))
(constraint (= (f #x098245ec8829de9a) #x098245ec8829de9b))
(constraint (= (f #x43ae887745ec8540) #x43ae887745ec8541))
(constraint (= (f #xbcdeb3a1e5866aee) #x864298bc34f32a23))
(check-synth)
(define-fun f_1 ((x (BitVec 64))) (BitVec 64) (ite (= (bvand #x0000000000000001 x) #x0000000000000000) (ite (= (bvurem x #x000000000000000b) #x0000000000000000) (bvxor #x0000000000000001 x) (ite (= (bvsrem x #x000000000000000b) #x0000000000000000) (bvnot (bvadd x x)) (ite (= (bvand #x000000000000000a x) #x0000000000000000) (ite (= (bvand #x0000000000000010 x) #x0000000000000000) (bvxor #x0000000000000001 x) (ite (= (bvurem x #x0000000000000003) #x0000000000000002) (bvxor #x0000000000000001 x) (bvnot (bvadd x x)))) (ite (= (bvurem x #x0000000000000007) #x0000000000000000) (bvnot (bvadd x x)) (ite (= (bvsrem x #x000000000000000b) #x0000000000000001) (bvxor #x0000000000000001 x) (ite (= (bvsrem x #x0000000000000009) #x0000000000000000) (ite (= (bvurem x #x0000000000000007) #x0000000000000001) (bvxor #x0000000000000001 x) (bvnot (bvadd x x))) (ite (= (bvsrem x #x0000000000000007) #x0000000000000001) (ite (= (bvurem x #x0000000000000005) #x0000000000000001) (bvxor #x0000000000000001 x) (bvnot (bvadd x x))) (ite (= (bvurem x #x0000000000000003) #x0000000000000001) (bvxor #x0000000000000001 x) (ite (= (bvurem x #x0000000000000005) #x0000000000000002) (bvxor #x0000000000000001 x) (ite (= (bvurem x #x0000000000000009) #x0000000000000002) (bvxor #x0000000000000001 x) (ite (= (bvand #x0000000000000007 x) #x0000000000000002) (bvnot (bvadd x x)) (ite (= (bvurem x #x0000000000000003) #x0000000000000002) (bvxor #x0000000000000001 x) (ite (= (bvurem x #x0000000000000009) #x0000000000000000) (ite (= (bvand #x0000000000000007 x) #x0000000000000000) (bvnot (bvadd x x)) (bvxor #x0000000000000001 x)) (bvnot (bvadd x x))))))))))))))) (ite (= (bvor #x0000000000000002 x) x) (ite (= (bvsrem x #x0000000000000009) #x0000000000000000) (ite (= (bvand #x0000000000000005 x) #x0000000000000001) (bvnot (bvadd x x)) (bvxor #x000000000000000f x)) (ite (= (bvand #x0000000000000005 x) #x0000000000000001) (ite (= (bvurem x #x000000000000000f) #x0000000000000000) (bvnot (bvadd x x)) (ite (= (bvand #x0000000000000010 x) #x0000000000000000) (bvxor #x0000000000000007 x) (ite (= (bvsrem x #x000000000000000b) #x0000000000000001) (bvnot (bvadd x x)) (ite (= (bvurem x #x0000000000000003) #x0000000000000002) (bvxor #x0000000000000007 x) (ite (= (bvand #x0000000000000008 x) #x0000000000000000) (bvxor #x0000000000000007 x) (bvnot (bvadd x x))))))) (bvnot (bvadd x x)))) (ite (= (bvsrem x #x000000000000000c) #x0000000000000001) (bvxor #x0000000000000003 x) (ite (= (bvurem x #x0000000000000007) #x0000000000000002) (bvxor #x0000000000000003 x) (ite (= (bvor #x000000000000000d x) x) (bvxor #x0000000000000003 x) (ite (= (bvurem x #x000000000000000f) #x0000000000000001) (bvxor #x0000000000000003 x) (ite (= (bvsrem x #x0000000000000007) #x0000000000000001) (bvxor #x0000000000000003 x) (ite (= (bvurem x #x000000000000000b) #x0000000000000000) (ite (= (bvurem x #x0000000000000003) #x0000000000000002) (bvnot (bvadd x x)) (bvxor #x0000000000000003 x)) (bvnot (bvadd x x)))))))))))
