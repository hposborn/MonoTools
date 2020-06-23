import sys
import logging
import os

import matplotlib as mpl
#Here, assuming this is server use, we make sure there's no X-server needed:
mpl.use('Agg')

#Making this nicer and therefore lower priority:
import os
os.nice(4)


##############################
#  FORCING LOGGING TO FILE:  #
##############################

if os.environ.get('MONOTOOLSPATH') is None:
    MonoData_savepath = os.path.join(os.path.dirname( __file__ ),'data')
else:
    MonoData_savepath = os.environ.get('MONOTOOLSPATH')
    
id_dic={'TESS':'TIC','tess':'TIC','Kepler':'KIC','kepler':'KIC','KEPLER':'KIC',
        'K2':'EPIC','k2':'EPIC','CoRoT':'CID','corot':'CID'}
    
ID=int(sys.argv[1])
mission=sys.argv[2]
x_file_loc=MonoData_savepath+'/'+id_dic[mission]+str(ID).zfill(11)
if not os.path.isdir(x_file_loc):
    os.mkdir(x_file_loc)

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
    filename=os.path.join(x_file_loc,id_dic[mission]+str(ID).zfill(11)+"_sysout.log"),
    filemode='a'
)

stdout_logger = logging.getLogger('STDOUT')
sl = StreamToLogger(stdout_logger, logging.INFO)
sys.stdout = sl

stderr_logger = logging.getLogger('STDERR')
sl = StreamToLogger(stderr_logger, logging.ERROR)
sys.stderr = sl

os.environ["THEANO_FLAGS"] = "device=cpu,floatX=float32,cxxflags = -fbracket-depth=1024,base_compiledir="+x_file_loc

####################################
#  GETTING ARGUMENTS AND RUNNING:  #
####################################

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

#Only running if we haven't already created a report:
if not os.path.exists(os.path.join(x_file_loc,id_dic[mission]+str(ID).zfill(11)+"_report.pdf")) or overwrite is not None:
    from MonoTools import MonoSearch
    try:
        
        outs=MonoSearch.MonoVetting(ID,mission,
                                    tcen=tcen,tdur=tdur,overwrite=overwrite,do_search=do_search,
                                    useL2=useL2,PL_ror_thresh=PL_ror_thresh,
                                    variable_llk_thresh=variable_llk_thresh,file_loc=file_loc,
                                    plot=plot,do_fit=do_fit,re_vet=re_vet,re_fit=re_fit, use_GP=False)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(e, exc_type, os.path.split(exc_tb.tb_frame.f_code.co_filename)[1], exc_tb.tb_lineno, ID,mission,"problem")

