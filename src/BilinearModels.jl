module BilinearModels

import SpecialFunctions
import TSVD

f(x) = SpecialFunctions.logabsgamma(x)[1]

g(A,k) = TSVD.tsvd(A,k)


end # module
