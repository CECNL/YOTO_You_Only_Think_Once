conda activate YOTO

# $dir = "psd_ASRICA_imagery"
# $dir = "psd_ASRICA_imagery_norm"
$dir = "psd_rtCLEEGN_imagery"
$arg = ""
# $dir = "psd_ASRICA_imagery_rmRest"
# $arg = "-b"

python .\plot_PSD.py -ic imagery_auditory fixation_fixation -ec imagery_visual --hide --save 1_auditory_only.png --savedir $dir/auditory_only/subject -ss 1 $arg
python .\plot_PSD.py -ic imagery_auditory fixation_fixation -ec imagery_visual --hide --save 2_auditory_only.png --savedir $dir/auditory_only/subject -ss 2 $arg
# Move-Item -Path "*auditory_only.png" -Destination "./auditory_only/subject"

python .\plot_PSD.py -ic imagery_visual fixation_fixation -ec imagery_auditory --hide --save 1_visual_only.png --savedir $dir/visual_only/subject -ss 1 $arg
python .\plot_PSD.py -ic imagery_visual fixation_fixation -ec imagery_auditory --hide --save 2_visual_only.png --savedir $dir/visual_only/subject -ss 2 $arg
# Move-Item -Path "*visual_only.png" -Destination "./visual_only/subject"

python .\plot_PSD.py -ic imagery_mix fixation_fixation --hide --save 1_mix.png --savedir $dir/mix/subject -ss 1 $arg
python .\plot_PSD.py -ic imagery_mix fixation_fixation --hide --save 2_mix.png --savedir $dir/mix/subject -ss 2 $arg
# Move-Item -Path "*mix.png" -Destination "./mix/subject"

# $dir = "psd_ASRICA_imagery_rmRest"
# $arg = "-b"

# python .\plot_PSD.py -ic imagery_auditory fixation_fixation -ec imagery_visual --hide --save 1_auditory_only.png --savedir $dir/auditory_only/subject -ss 1 $arg
# python .\plot_PSD.py -ic imagery_auditory fixation_fixation -ec imagery_visual --hide --save 2_auditory_only.png --savedir $dir/auditory_only/subject -ss 2 $arg
# # Move-Item -Path "*auditory_only.png" -Destination "./auditory_only/subject"

# python .\plot_PSD.py -ic imagery_visual fixation_fixation -ec imagery_auditory --hide --save 1_visual_only.png --savedir $dir/visual_only/subject -ss 1 $arg
# python .\plot_PSD.py -ic imagery_visual fixation_fixation -ec imagery_auditory --hide --save 2_visual_only.png --savedir $dir/visual_only/subject -ss 2 $arg
# # Move-Item -Path "*visual_only.png" -Destination "./visual_only/subject"

# python .\plot_PSD.py -ic imagery_mix fixation_fixation --hide --save 1_mix.png --savedir $dir/mix/subject -ss 1 $arg
# python .\plot_PSD.py -ic imagery_mix fixation_fixation --hide --save 2_mix.png --savedir $dir/mix/subject -ss 2 $arg
# # Move-Item -Path "*mix.png" -Destination "./mix/subject"