rm(list=ls())
pet_parcels <- read.csv("2_BrainAge/PET_MRI_age_rev1/data/ADNI/CN/ADNI_PET_CN_Sch_Tian_1mm_parcels.csv")
mri_parcels <- read.csv("2_BrainAge/PET_MRI_age_rev1/data/ADNI/CN/ADNI_MRI_Sch_Tian_1mm_parcels.csv")

# coef_pet$region[(coef_pet$avg < -0.5)]
pet_negweight <- c("X17Networks_RH_SalVentAttnB_PFCmp_1", "X17Networks_RH_TempPar_4",
                   "NAc.rh", "GP.rh")
pet_posweight <- c("X17Networks_RH_DorsAttnA_TempOcc_1", "X17Networks_RH_DefaultA_PFCm_1",
                   "X17Networks_RH_DefaultA_PFCm_2", "X17Networks_RH_DefaultC_Rsp_1", "CAU.rh"    )
mri_negweight <- c("X17Networks_LH_ContA_PFCl_2", "X17Networks_RH_VisPeri_StriCal_1",
                   "X17Networks_RH_VisPeri_ExStrSup_3", "NAc.rh", "GP.rh",
                   "PUT.rh", "CAU.rh")
mri_posweight <- c("X17Networks_LH_SalVentAttnA_ParMed_1",
                   "X17Networks_LH_DefaultB_PFCd_2",
                   "X17Networks_LH_DefaultB_PFCd_4",     
                   "X17Networks_RH_SalVentAttnA_Ins_1",
                   "X17Networks_RH_ContB_IPL_2", "HIP.rh")
for (p in pet_negweight){
  print(p)
  print(cor.test(pet_parcels$age, pet_parcels[,p]))
}

for (p in pet_posweight){
  print(p)
  print(cor.test(pet_parcels$age, pet_parcels[,p]))
}

for (m in mri_negweight){
  print(m)
  print(cor.test(mri_parcels$age, mri_parcels[,m]))
}

for (m in mri_posweight){
  print(m)
  print(cor.test(mri_parcels$age, mri_parcels[,m]))
}
