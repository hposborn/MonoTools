import sys

import logging
import os
import traceback

import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='Run MonoTools.MonoSearch.MonoVetting on a single target')

import matplotlib as mpl
#Here, assuming this is server use, we make sure there's no X-server needed:
mpl.use('Agg')

#Making this nicer and therefore lower priority:
import os
os.nice(4)

log=False
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

    
if log:
    #Functions needed to make this stream logging work:
    sys.stdout.isatty = lambda: False

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

if "THEANO_FLAGS" in os.environ:
    os.environ["THEANO_FLAGS"]+=", base_compiledir="+x_file_loc
else:
    os.environ["THEANO_FLAGS"]="base_compiledir="+x_file_loc

####################################
#  GETTING ARGUMENTS AND RUNNING:  #
####################################
parser.add_argument('id', type=int, help='ID of candidate (e.g. in TIC, KIC, EPIC, etc)', default=None, nargs=1)
parser.add_argument('mission', type=str, help='name of mission', default=None, nargs=1)

parser.add_argument('--tcen', type=float, help='Central transit time', default=None, nargs=1)
parser.add_argument('--tdur', type=float, help='Transit duration in days', default=None, nargs=1)
parser.add_argument('--overwrite', type=str, help='which steps to overwrite', default=None, nargs=1)
parser.add_argument('--do_search', type=bool, help='Perform search for all possible monotransits?', default=True, nargs=1)
parser.add_argument('--useL2', type=bool, help='Include second light in transit modelling (e.g. for EBs)?', default=False, nargs=1)
parser.add_argument('--PL_ror_thresh', type=float, help='Upper bound in Rp/Rs to count as planets', default=0.2, nargs=1)
parser.add_argument('--variable_llk_thresh', type=float, help='Likelihood threshold for transit over variability', default=5, nargs=1)
parser.add_argument('--file_loc', type=str, help='Location to store outputs. Otherwise uses $MONOTOOLSPATH', default=None, nargs=1)
parser.add_argument('--plot', type=bool, help='Perform search for all possible monotransits?', default=True, nargs=1)
parser.add_argument('--do_fit', type=bool, help='Perform search for all possible monotransits?', default=True, nargs=1)
parser.add_argument('--use_GP', type=bool, help='Model using Celerite GP?', default=False, nargs=1)
'''tcen=float(sys.argv[3]) if len(sys.argv)>3 else None
tdur=float(sys.argv[4]) if len(sys.argv)>4 else None
overwrite=sys.argv[5] if len(sys.argv)>5 else None
do_search=bool(sys.argv[6]) if len(sys.argv)>6 else True
useL2=bool(sys.argv[7]) if len(sys.argv)>7 else False
PL_ror_thresh=float(sys.argv[8]) if len(sys.argv)>8 else 0.2
variable_llk_thresh=float(sys.argv[9]) if len(sys.argv)>9 else 5
file_loc=sys.argv[10] if len(sys.argv)>10 else None
plot=bool(sys.argv[11]) if len(sys.argv)>11 else True
do_fit=bool(sys.argv[12]) if len(sys.argv)>12 else True
'''

args = parser.parse_args()


#Only running if we haven't already created a report:
if not os.path.exists(os.path.join(x_file_loc,id_dic[mission]+str(ID).zfill(11)+"_report.pdf")) or args.overwrite is not None:
    from MonoTools import MonoSearch
    #try:
    outs=MonoSearch.MonoVetting(ID,mission,
                                tcen=args.tcen,tdur=args.tdur,overwrite=args.overwrite,do_search=args.do_search,
                                useL2=args.useL2,PL_ror_thresh=args.PL_ror_thresh,
                                variable_llk_thresh=args.variable_llk_thresh,file_loc=args.file_loc,
                                plot=args.plot, do_fit=args.do_fit, use_GP=args.use_GP)
    #except Exception as e:
    #    #traceback.print_exc()
    #    exc_type, exc_obj, exc_tb = sys.exc_info()
    #    print(e, exc_type, os.path.split(exc_tb.tb_frame.f_code.co_filename)[1], exc_tb.tb_lineno, ID,mission,"problem")
        
