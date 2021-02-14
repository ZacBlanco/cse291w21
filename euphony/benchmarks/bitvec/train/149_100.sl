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
(constraint (= (f #x9b99e16b92a597e5) #x0000000000000002))
(constraint (= (f #x3ad27812234de63e) #xfffffc52d87eddcb))
(constraint (= (f #x680eace41c800002) #xfffff97f1531be37))
(constraint (= (f #x623b46ded310e2dd) #x0000000000000002))
(constraint (= (f #x2c1b099e2d210b86) #xfffffd3e4f661d2d))
(constraint (= (f #x8a3c021655e96c3b) #x0000000000000002))
(constraint (= (f #x6d3ab61900b245ec) #xfffff92c549e6ff4))
(constraint (= (f #xa66146857e35a41b) #x0000000000000002))
(constraint (= (f #x16d1d9210eb0561a) #xfffffe92e26def14))
(constraint (= (f #x233764c661bc2e9b) #x0000000000000002))
(constraint (= (f #xd3671d2de78b6425) #x0000000000000002))
(constraint (= (f #x4e26ab3d2492a5ed) #x0000000000000002))
(constraint (= (f #x06e979c62dee6db6) #xffffff9168639d21))
(constraint (= (f #x6d06e6c967ebead2) #xfffff92f91936981))
(constraint (= (f #x06d93629d6184c92) #xffffff926c9d629e))
(constraint (= (f #x2a1a826515cee2e9) #x0000000000000002))
(constraint (= (f #xb84b800867665a92) #xfffff47b47ff7989))
(constraint (= (f #x6856c1a2a75e6800) #xfffff97a93e5d58a))
(constraint (= (f #x52eda2adc8469dec) #xfffffad125d5237b))
(constraint (= (f #x50e963dd8869e96a) #xfffffaf169c22779))
(constraint (= (f #x55791335d20039ec) #xfffffaa86ecca2df))
(constraint (= (f #x9ce86816ce435ce2) #xfffff631797e931b))
(constraint (= (f #xab40b1cc8204eecb) #x0000000000000002))
(constraint (= (f #x1c6d154b1427e816) #xfffffe392eab4ebd))
(constraint (= (f #x3069832e39b4e3e7) #x0000000000000002))
(constraint (= (f #x722c5720ee097c6a) #xfffff8dd3a8df11f))
(constraint (= (f #xd72ed985547b2ad9) #x0000000000000002))
(constraint (= (f #x446ce28bca59250e) #xfffffbb931d7435a))
(constraint (= (f #x1c1ac75e57e2749b) #x0000000000000002))
(constraint (= (f #xe9512a583727d10b) #x0000000000000002))
(constraint (= (f #x6a0ce862e96ae4e9) #x0000000000000002))
(constraint (= (f #x1e75a8dec45a6945) #x0000000000000002))
(constraint (= (f #xead625e0d2e2419b) #x0000000000000002))
(constraint (= (f #xd9ee33bb65074a2c) #xfffff2611cc449af))
(constraint (= (f #xe01db0c3131addbb) #x0000000000000002))
(constraint (= (f #x13ead9d7a359854d) #x0000000000000002))
(constraint (= (f #x2a1eb3583adcd213) #x0000000000000002))
(constraint (= (f #xdbc110adebca12d5) #x0000000000000002))
(constraint (= (f #x170b7495e3ece484) #xfffffe8f48b6a1c1))
(constraint (= (f #xdd130418e9ec484c) #xfffff22ecfbe7161))
(constraint (= (f #x0de1bae53615b367) #x0000000000000002))
(constraint (= (f #xeea8952929d50e94) #xfffff11576ad6d62))
(constraint (= (f #xe62842e54434710b) #x0000000000000002))
(constraint (= (f #xeb277de82e1ae412) #xfffff14d88217d1e))
(constraint (= (f #x8d8ae27ecd5381ac) #xfffff72751d8132a))
(constraint (= (f #x4435ea51bea53d89) #x0000000000000002))
(constraint (= (f #x93665dd63042930c) #xfffff6c99a229cfb))
(constraint (= (f #xbe3819ae932cdb18) #xfffff41c7e6516cd))
(constraint (= (f #xece0550a5eac5ae1) #x0000000000000002))
(constraint (= (f #x17b9b413b82528e2) #xfffffe8464bec47d))
(constraint (= (f #xec5e3accd55d23d4) #xfffff13a1c5332aa))
(constraint (= (f #xd225ca0a8ea763e9) #x0000000000000002))
(constraint (= (f #xbee00dc855cecc0e) #xfffff411ff237aa3))
(constraint (= (f #x5e811d843178590c) #xfffffa17ee27bce8))
(constraint (= (f #xa9c07e6e8baecce8) #xfffff563f8191745))
(constraint (= (f #x63009cb31a85e708) #xfffff9cff634ce57))
(constraint (= (f #x75882ede0eb1ad84) #xfffff8a77d121f14))
(constraint (= (f #x222514714236eeca) #xfffffdddaeb8ebdc))
(constraint (= (f #x03cae881e2684b7b) #x0000000000000002))
(constraint (= (f #x7ee87ad9ebe4e456) #xfffff81178526141))
(constraint (= (f #x6c198a7e463627ad) #x0000000000000002))
(constraint (= (f #xa67ed68add8c7dae) #xfffff59812975227))
(constraint (= (f #x44557138322e4b7c) #xfffffbbaa8ec7cdd))
(constraint (= (f #xee46b970eaa9b2d1) #x0000000000000002))
(constraint (= (f #xe7c61ce2e9c07de7) #x0000000000000002))
(constraint (= (f #xa27b51ceaedd6d79) #x0000000000000002))
(constraint (= (f #x1774e5dc14edcb76) #xfffffe88b1a23eb1))
(constraint (= (f #x079e9c3858acad5e) #xffffff86163c7a75))
(constraint (= (f #x09810eed0e0b3e75) #x0000000000000002))
(constraint (= (f #x0c13577dd50ad688) #xffffff3eca8822af))
(constraint (= (f #xa43ebb830e9638d4) #xfffff5bc1447cf16))
(constraint (= (f #x7699e2853edd4bd0) #xfffff89661d7ac12))
(constraint (= (f #x2840036892e6c329) #x0000000000000002))
(constraint (= (f #xbc26705ee5154da1) #x0000000000000002))
(constraint (= (f #x4ca5ee918305ecb7) #x0000000000000002))
(constraint (= (f #xc84398ae64ce3e53) #x0000000000000002))
(constraint (= (f #xc2e8406c73746dbd) #x0000000000000002))
(constraint (= (f #x00aed06b5ebb3b1e) #xfffffff512f94a14))
(constraint (= (f #xccc98e62571b0b19) #x0000000000000002))
(constraint (= (f #xc4ee554de23d33d3) #x0000000000000002))
(constraint (= (f #x36a61556c8854b40) #xfffffc959eaa9377))
(constraint (= (f #x8eeee4e21423121a) #xfffff71111b1debd))
(constraint (= (f #xc8a2e2c041ee291d) #x0000000000000002))
(constraint (= (f #x8ddd48ec7e2d99e6) #xfffff7222b71381d))
(constraint (= (f #x7c4e07d1ec0d103e) #xfffff83b1f82e13f))
(constraint (= (f #xce14c5d341005084) #xfffff31eb3a2cbef))
(constraint (= (f #x1be348bc84521b8a) #xfffffe41cb7437ba))
(constraint (= (f #x5649e031256860ea) #xfffffa9b61fceda9))
(constraint (= (f #xb333368eb5e7a892) #xfffff4cccc9714a1))
(constraint (= (f #x3452ed79dc460ee0) #xfffffcbad128623b))
(constraint (= (f #x9d1ca4b3a8aa31ce) #xfffff62e35b4c575))
(constraint (= (f #x16b6244222677dd9) #x0000000000000002))
(constraint (= (f #x65497cb31cce9747) #x0000000000000002))
(constraint (= (f #xd93be864748a07b8) #xfffff26c4179b8b7))
(constraint (= (f #x53aa0c7364e71676) #xfffffac55f38c9b1))
(constraint (= (f #x5ce299ee17352e74) #xfffffa31d6611e8c))
(constraint (= (f #x258bcc92d3127d59) #x0000000000000002))
(constraint (= (f #x5e7557a98d4e3861) #x0000000000000002))
(constraint (= (f #x201a858a7cb2e22e) #xfffffdfe57a75834))
(constraint (= (f #xe737789436368791) #x0000000000000002))
(check-synth)
(define-fun f_1 ((x (BitVec 64))) (BitVec 64) (ite (= (bvor #x0000000000000001 x) x) #x0000000000000002 (bvnot (bvudiv (bvlshr x #x0000000000000010) #x0000000000000010))))
