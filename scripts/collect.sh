# env_name=Putbullet_vaseintoBowlDiverseTwoDistractors-v0
# env_name=Putglass_half_gallonintoBowlDiverseTwoDistractors-v0
# env_name=PutT_cupintoBowlDiverseTwoDistractors-v0
# env_name=Putcolunnade_topintoBowlDiverseTwoDistractors-v0
# env_name=PutAerointoBowl-v0
# env_name=Putbeehive_funnelintoBowl-v0
# env_name=Putpacifier_vaseintoBowl-v0
# env_name=Putcrooked_lid_trash_canlintoBowl-v0
# env_name=Puttongue_chairintoBowl-v0
# env_name=Putvintage_canoeintoBowl-v0
env_name=PutBallintoBowl-v0
python scripts/scripted_collect.py \
-e $env_name \
-pl pickplace \
-n 250 \
-t 40 \
-d /nfs/kun2/users/paulzhou/cog_collect \
-a place_success_target \
--noise 0.1
