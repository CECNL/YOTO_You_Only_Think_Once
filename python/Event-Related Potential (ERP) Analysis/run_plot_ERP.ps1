conda activate YOTO

$dir = "erp_ASRICA_stimuli"
$i1 = -1
$i2 = 0.6
$arg = ("")

# $dir = "erp_ASRICA_imagery"
# $i1 = -1
# $i2 = 0.6
# $arg = ("--eventOffset", "1")

python .\plot_ERP.py -ic stimuli_auditory -ec stimuli_visual --hide --save auditory_only.png --savedir $dir/auditory_only/subject --interval $i1 $i2 @arg
# python .\plot_ERP.py -ic stimuli_auditory -ec stimuli_visual --hide --save 2_auditory_only.png --savedir $dir/auditory_only/subject -ss 2 --interval $i1 $i2

python .\plot_ERP.py -ic stimuli_visual -ec stimuli_auditory --hide --save visual_only.png --savedir $dir/visual_only/subject --interval $i1 $i2 @arg
python .\plot_ERP.py -it stimuli_21 --hide --save visual_21.png --savedir $dir/visual_trigger/subject --interval $i1 $i2 @arg
python .\plot_ERP.py -it stimuli_22 --hide --save visual_22.png --savedir $dir/visual_trigger/subject --interval $i1 $i2 @arg
python .\plot_ERP.py -it stimuli_21 stimuli_22 --hide --save visual_face.png --savedir $dir/visual_trigger/subject --interval $i1 $i2 @arg
python .\plot_ERP.py -it stimuli_23 --hide --save visual_23.png --savedir $dir/visual_trigger/subject --interval $i1 $i2 @arg
# python .\plot_ERP.py -ic stimuli_visual -ec stimuli_auditory --hide --save 2_visual_only.png --savedir $dir/visual_only/subject -ss 2 --interval $i1 $i2 @arg
# python .\plot_ERP.py -it stimuli_21 --hide --save 2_visual_21.png --savedir $dir/visual_trigger/subject -ss 2 --interval $i1 $i2 @arg
# python .\plot_ERP.py -it stimuli_22 --hide --save 2_visual_22.png --savedir $dir/visual_trigger/subject -ss 2 --interval $i1 $i2 @arg
# python .\plot_ERP.py -it stimuli_21 stimuli_22 --hide --save 2_visual_face.png --savedir $dir/visual_trigger/subject -ss 2 --interval $i1 $i2 @arg
# python .\plot_ERP.py -it stimuli_23 --hide --save 2_visual_23.png --savedir $dir/visual_trigger/subject -ss 2 --interval $i1 $i2 @arg
# Move-item -Path "*visual_only.png" -Destination "./visual_only/subject"

python .\plot_ERP.py -ic stimuli_mix --hide --save mix.png --savedir $dir/mix/subject --interval $i1 $i2 @arg
# python .\plot_ERP.py -ic stimuli_mix --hide --save 2_mix.png --savedir $dir/mix/subject -ss 2 --interval $i1 $i2 @arg
# Move-item -Path "*mix.png" -Destination "./mix/subject"


# -----------------plot channel-------------------

python .\plot_ERP.py -ic stimuli_auditory -ec stimuli_visual --hide --save auditory_only.png --savedir $dir/auditory_only/channel --type channel --interval $i1 $i2 @arg
# python .\plot_ERP.py -ic stimuli_auditory -ec stimuli_visual --hide --save 2_auditory_only.png --savedir $dir/auditory_only/channel --type channel -ss 2 --interval $i1 $i2 @arg
# Move-item -Path "*auditory_only.png" -Destination "./auditory_only/channel"

python .\plot_ERP.py -ic stimuli_visual -ec stimuli_auditory --hide --save visual_only.png --savedir $dir/visual_only/channel --type channel --interval $i1 $i2 @arg
python .\plot_ERP.py -it stimuli_21 --hide --save visual_21.png --savedir $dir/visual_trigger/channel --type channel --interval $i1 $i2 @arg
python .\plot_ERP.py -it stimuli_22 --hide --save visual_22.png --savedir $dir/visual_trigger/channel --type channel --interval $i1 $i2 @arg
python .\plot_ERP.py -it stimuli_21 stimuli_22 --hide --save visual_face.png --savedir $dir/visual_trigger/channel --type channel --interval $i1 $i2 @arg
python .\plot_ERP.py -it stimuli_23 --hide --save visual_23.png --savedir $dir/visual_trigger/channel --type channel --interval $i1 $i2 @arg
# python .\plot_ERP.py -ic stimuli_visual -ec stimuli_auditory --hide --save 2_visual_only.png --savedir $dir/visual_only/channel --type channel -ss 2 --interval $i1 $i2 @arg
# python .\plot_ERP.py -it stimuli_21 --hide --save 2_visual_21.png --savedir $dir/visual_trigger/channel --type channel -ss 2 --interval $i1 $i2 @arg
# python .\plot_ERP.py -it stimuli_22 --hide --save 2_visual_22.png --savedir $dir/visual_trigger/channel --type channel -ss 2 --interval $i1 $i2 @arg
# python .\plot_ERP.py -it stimuli_23 --hide --save 2_visual_23.png --savedir $dir/visual_trigger/channel --type channel -ss 2 --interval $i1 $i2 @arg
# Move-item -Path "*visual_only.png" -Destination "./visual_only/channel"
python .\plot_ERP.py -ic stimuli_mix --hide --save mix.png --savedir $dir/mix/channel --type channel --interval $i1 $i2 @arg
# python .\plot_ERP.py -ic stimuli_mix --hide --save 2_mix.png --savedir $dir/mix/channel --type channel -ss 2 --interval $i1 $i2 @arg
# Move-item -Path "*mix.png" -Destination "./mix/channel"
