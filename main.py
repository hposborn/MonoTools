import sys

#Making this nicer and therefore lower priority:
import os
os.nice(4)

ID=int(sys.argv[1])
mission=sys.argv[2]
print(sys.argv, len(sys.argv))
tcen=float(sys.argv[3]) if len(sys.argv)>3 else None
tdur=float(sys.argv[4]) if len(sys.argv)>4 else None
overwrite=sys.argv[5] if len(sys.argv)>5 else None
do_search=bool(sys.argv[6]) if len(sys.argv)>6 else True
useL2=bool(sys.argv[7]) if len(sys.argv)>7 else False
PL_ror_thresh=float(sys.argv[8]) if len(sys.argv)>8 else 0.2
variable_llk_thresh=float(sys.argv[9]) if len(sys.argv)>9 else 5
file_loc=sys.argv[10] if len(sys.argv)>10 else None
plot=bool(sys.argv[11]) if len(sys.argv)>11 else True
do_fit=bool(sys.argv[12]) if len(sys.argv)>12 else True
re_vet=bool(sys.argv[13]) if len(sys.argv)>13 else True
re_fit=bool(sys.argv[14]) if len(sys.argv)>14 else True

from MonoTools import MonoSearch

out=MonoSearch.MonoVetting(ID,mission,
                            tcen=tcen,tdur=tdur,overwrite=overwrite,do_search=do_search,
                            useL2=useL2,PL_ror_thresh=PL_ror_thresh,variable_llk_thresh=variable_llk_thresh,file_loc=file_loc,
                            plot=plot,do_fit=do_fit,re_vet=re_vet,re_fit=re_fit, use_GP=False)
