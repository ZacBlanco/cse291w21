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
(constraint (= (f #x0e1ac4e66a0441e1) #x00000e1ac4e66a04))
(constraint (= (f #x397870594d0477b5) #x0000397870594d04))
(constraint (= (f #x82786da0e1eee184) #x04f0db41c3ddc30a))
(constraint (= (f #x5530b41a4e56992c) #xaa6168349cad325a))
(constraint (= (f #x245e7eabd08ed598) #x48bcfd57a11dab32))
(constraint (= (f #xebe13ed56489e428) #xd7c27daac913c852))
(constraint (= (f #x17730c5a51227bb1) #x000017730c5a5122))
(constraint (= (f #x1c662b457e1e3a3c) #x38cc568afc3c747a))
(constraint (= (f #xc7e474e833c3ce0c) #x8fc8e9d067879c1a))
(constraint (= (f #xb410e06ddc815d2c) #x6821c0dbb902ba5a))
(constraint (= (f #x2d3b01eb3c7aed50) #x5a7603d678f5daa2))
(constraint (= (f #xe06e247e990a5944) #xc0dc48fd3214b28a))
(constraint (= (f #x279906418b8ebb49) #x0000279906418b8e))
(constraint (= (f #x3cdde14e71b53686) #x79bbc29ce36a6d0e))
(constraint (= (f #xb585c7319ece1b09) #x0000b585c7319ece))
(constraint (= (f #xe0ede2deccb8a283) #x0000e0ede2deccb8))
(constraint (= (f #x023c748c597ea24c) #x0478e918b2fd449a))
(constraint (= (f #x87c70397033eebae) #x0f8e072e067dd75e))
(constraint (= (f #x2ee7e9e6d0721416) #x5dcfd3cda0e4282e))
(constraint (= (f #x10436ca21663a174) #x2086d9442cc742ea))
(constraint (= (f #xb60547de499d7d8e) #x6c0a8fbc933afb1e))
(constraint (= (f #x578952255a9bd9e6) #xaf12a44ab537b3ce))
(constraint (= (f #x5e0a4e2313212ecb) #x00005e0a4e231321))
(constraint (= (f #xa2ce3e40e612e7c4) #x459c7c81cc25cf8a))
(constraint (= (f #xc787dbce2dd5b0e3) #x0000c787dbce2dd5))
(constraint (= (f #x948e2ec4aa34527d) #x0000948e2ec4aa34))
(constraint (= (f #x83783133b32c7dac) #x06f062676658fb5a))
(constraint (= (f #x1e2e93c8579a59e5) #x00001e2e93c8579a))
(constraint (= (f #x43c40eeda618cde1) #x000043c40eeda618))
(constraint (= (f #x2cee103de9e6de3a) #x59dc207bd3cdbc76))
(constraint (= (f #x9d862a8c0c7ee695) #x00009d862a8c0c7e))
(constraint (= (f #xdd931434e4a7e611) #x0000dd931434e4a7))
(constraint (= (f #xae4bda570e647193) #x0000ae4bda570e64))
(constraint (= (f #xeec6681edeee3957) #x0000eec6681edeee))
(constraint (= (f #xe3aea9c4180eacbd) #x0000e3aea9c4180e))
(constraint (= (f #xd2e0470db7a9090c) #xa5c08e1b6f52121a))
(constraint (= (f #x9d1519588ed0b675) #x00009d1519588ed0))
(constraint (= (f #xe772b96d710259ed) #x0000e772b96d7102))
(constraint (= (f #xe63dd30120ddd130) #xcc7ba60241bba262))
(constraint (= (f #xedeb451598d1eedc) #xdbd68a2b31a3ddba))
(constraint (= (f #x59a9ee1869286dac) #xb353dc30d250db5a))
(constraint (= (f #x116ee311d36714ac) #x22ddc623a6ce295a))
(constraint (= (f #x2dbe7a299c0be142) #x5b7cf4533817c286))
(constraint (= (f #x11887415c826010d) #x000011887415c826))
(constraint (= (f #x35d0ab6de8e238e1) #x000035d0ab6de8e2))
(constraint (= (f #xe2dda4102a6edc93) #x0000e2dda4102a6e))
(constraint (= (f #x9ed706bdd831ac9c) #x3dae0d7bb063593a))
(constraint (= (f #xa0e110605eb2b3e9) #x0000a0e110605eb2))
(constraint (= (f #x0441ec9a449c568e) #x0883d9348938ad1e))
(constraint (= (f #xeb5a0a155e99e855) #x0000eb5a0a155e99))
(constraint (= (f #x7bbe380cad1d2a3e) #xf77c70195a3a547e))
(constraint (= (f #x56db8e4c966480a6) #xadb71c992cc9014e))
(constraint (= (f #xceaec6e1be17e60e) #x9d5d8dc37c2fcc1e))
(constraint (= (f #x2a74077d67d2bee6) #x54e80efacfa57dce))
(constraint (= (f #x4e579a5e1684acee) #x9caf34bc2d0959de))
(constraint (= (f #x9714e118db24eb6c) #x2e29c231b649d6da))
(constraint (= (f #x52684dc3200bd2bd) #x000052684dc3200b))
(constraint (= (f #x2e315b2bb09427b5) #x00002e315b2bb094))
(constraint (= (f #xeaed342c819b9ec0) #xd5da685903373d82))
(constraint (= (f #x0177e1664bb2a173) #x00000177e1664bb2))
(constraint (= (f #x7ad636bbe3808959) #x00007ad636bbe380))
(constraint (= (f #xa4dbe270526e4c40) #x49b7c4e0a4dc9882))
(constraint (= (f #xe7b9e949dd2c9e8c) #xcf73d293ba593d1a))
(constraint (= (f #x221eb23a45e90b68) #x443d64748bd216d2))
(constraint (= (f #x2159a1d1b2cee632) #x42b343a3659dcc66))
(constraint (= (f #xee788e4aeddeebe2) #xdcf11c95dbbdd7c6))
(constraint (= (f #xad89234d4ea4eaa0) #x5b12469a9d49d542))
(constraint (= (f #x51eb9e6ce60e1213) #x000051eb9e6ce60e))
(constraint (= (f #x8d1e5a86c56b4eea) #x1a3cb50d8ad69dd6))
(constraint (= (f #x36de0688755e73d4) #x6dbc0d10eabce7aa))
(constraint (= (f #x7b681bdc4a16adc1) #x00007b681bdc4a16))
(constraint (= (f #xe0b760d860aa83aa) #xc16ec1b0c1550756))
(constraint (= (f #xe2552b1eb73ebbda) #xc4aa563d6e7d77b6))
(constraint (= (f #xdde54a702b780c5c) #xbbca94e056f018ba))
(constraint (= (f #x0ce49b82610b5a16) #x19c93704c216b42e))
(constraint (= (f #x7aa9eec0b4034c4a) #xf553dd8168069896))
(constraint (= (f #x4c690e5b5abb8ee4) #x98d21cb6b5771dca))
(constraint (= (f #x05b7ea1b0955b37e) #x0b6fd43612ab66fe))
(constraint (= (f #x9b7209e1ec58e65e) #x36e413c3d8b1ccbe))
(constraint (= (f #xa6ee1760e2775e5e) #x4ddc2ec1c4eebcbe))
(constraint (= (f #x26904a82eb66e42b) #x000026904a82eb66))
(constraint (= (f #x4a3717bb887e77e4) #x946e2f7710fcefca))
(constraint (= (f #xb676b9eb0d72b118) #x6ced73d61ae56232))
(constraint (= (f #x2e911c81e555c229) #x00002e911c81e555))
(constraint (= (f #xe040e8a625e2829a) #xc081d14c4bc50536))
(constraint (= (f #x758beabb0803ade8) #xeb17d57610075bd2))
(constraint (= (f #xa366b702e0de05a6) #x46cd6e05c1bc0b4e))
(constraint (= (f #x267a8ee6e60ae4ee) #x4cf51dcdcc15c9de))
(constraint (= (f #x897ad727dd44a975) #x0000897ad727dd44))
(constraint (= (f #x5ec0d113ab23841c) #xbd81a2275647083a))
(constraint (= (f #xee01bc0d3e2beb34) #xdc03781a7c57d66a))
(constraint (= (f #xb9b16e9abe355e48) #x7362dd357c6abc92))
(constraint (= (f #x08e67bbe0bb3eb61) #x000008e67bbe0bb3))
(constraint (= (f #x2972569de37e8b45) #x00002972569de37e))
(constraint (= (f #x7e5e9ae14182a8b2) #xfcbd35c283055166))
(constraint (= (f #xa08ce2e451753512) #x4119c5c8a2ea6a26))
(constraint (= (f #x6909b5d5d55a8215) #x00006909b5d5d55a))
(constraint (= (f #x09a2db5eaeddc8b5) #x000009a2db5eaedd))
(constraint (= (f #xcc1bda27c33c6b79) #x0000cc1bda27c33c))
(constraint (= (f #x8dbd6ee523ab8336) #x1b7addca4757066e))
(check-synth)
(define-fun f_1 ((x (BitVec 64))) (BitVec 64) (ite (= (bvor #x0000000000000001 x) x) (bvlshr x #x0000000000000010) (bvxor (bvadd x x) #x0000000000000002)))
