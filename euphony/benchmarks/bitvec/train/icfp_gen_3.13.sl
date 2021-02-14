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
(constraint (= (f #x9B3E7E38FC7572DE) #x367CFC71F8EAE5BC))
(constraint (= (f #xE3DD279FA29CDFDC) #xC7BA4F3F4539BFB8))
(constraint (= (f #x730B59A5800DF8D6) #xE616B34B001BF1AC))
(constraint (= (f #x8820034AC708C9F0) #x104006958E1193E0))
(constraint (= (f #x912E31F3B85CC198) #x225C63E770B98330))
(constraint (= (f #xFFFFFFFFFFFFFFF8) #xFFFFFFFFFFFFFFF0))
(constraint (= (f #x131A0D018B31676F) #x021008010220464E))
(constraint (= (f #x50EF7B7137CE767D) #x00CE7260278C6478))
(constraint (= (f #x7ED0A5982B8C74B3) #x7C80011003086022))
(constraint (= (f #x83A9BBACEDB84F05) #x03013308C9300E00))
(constraint (= (f #xA1D9609AC1757067) #x0190401080606046))
(constraint (= (f #xFFFFFFFFFFFFFFFA) #xFFFFFFFFFFFFFFF4))
(constraint (= (f #xB800000000000001) #x3000000000000000))
(constraint (= (f #xC000000000000001) #x8000000000000000))
(constraint (= (f #x5000000000000001) #x0000000000000000))
(constraint (= (f #x0800000000000001) #x0000000000000000))
(constraint (= (f #xD800000000000001) #x9000000000000000))
(constraint (= (f #xFFFFFFFFFFFFFFF9) #xFFFFFFFFFFFFFFF0))
(constraint (= (f #xFFFFFFFFFFFFFFFB) #xFFFFFFFFFFFFFFFB))
(constraint (= (f #xC99C4F7B6B0F1223) #x81180E72420E0002))
(constraint (= (f #x9DBFD92DAD81DB12) #x3B7FB25B5B03B624))
(constraint (= (f #x9D27541F506BA656) #x3A4EA83EA0D74CAC))
(constraint (= (f #xAD2DC906FC4DCD77) #x08098004F8098866))
(constraint (= (f #x6A39D7F941A76A7B) #x403187F001064072))
(constraint (= (f #xBABB56831688111B) #x3032040204000012))
(constraint (= (f #x59ABB7CB73A84DB7) #x1103278263000926))
(constraint (= (f #xC5B36806C1DAA1B5) #x8122400481900120))
(constraint (= (f #x5A75709FF0990853) #x1060601FE0100002))
(constraint (= (f #xE5A74CEE5A90AEF3) #xC10608CC10000CE2))
(constraint (= (f #xFFFFFFFFFFFFFFFB) #xFFFFFFFFFFFFFFFB))
(constraint (= (f #xFFFFFFFFFFFFFFFA) #xFFFFFFFFFFFFFFF4))
(constraint (= (f #xFFFFFFFFFFFFFFF9) #xFFFFFFFFFFFFFFF0))
(constraint (= (f #x1000000000000001) #x0000000000000000))
(check-synth)
(define-fun f_1 ((x (BitVec 64))) (BitVec 64) (ite (= (bvnot #x0000000000000004) x) x (ite (= (bvor #x0000000000000001 x) x) (bvand (bvadd x x) x) (bvadd x x))))
