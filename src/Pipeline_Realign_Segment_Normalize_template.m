%-----------------------------------------------------------------------
% Job saved on 23-Mar-2021 10:28:01 by cfg_util (rev $Rev: 7345 $)
% spm SPM - SPM12 (7771)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
matlabbatch{1}.spm.spatial.realign.estimate.data = {
                                                    {
                                                    'REPLACE_FDG_IMAGE,47'
                                                    'REPLACE_FDG_IMAGE,48'
                                                    'REPLACE_FDG_IMAGE,49'
                                                    'REPLACE_FDG_IMAGE,50'
                                                    'REPLACE_FDG_IMAGE,51'
                                                    'REPLACE_FDG_IMAGE,52'
                                                    }
                                                    }';
matlabbatch{1}.spm.spatial.realign.estimate.eoptions.quality = 0.9;
matlabbatch{1}.spm.spatial.realign.estimate.eoptions.sep = 4;
matlabbatch{1}.spm.spatial.realign.estimate.eoptions.fwhm = 5;
matlabbatch{1}.spm.spatial.realign.estimate.eoptions.rtm = 0;
matlabbatch{1}.spm.spatial.realign.estimate.eoptions.interp = 2;
matlabbatch{1}.spm.spatial.realign.estimate.eoptions.wrap = [0 0 0];
matlabbatch{1}.spm.spatial.realign.estimate.eoptions.weight = '';
matlabbatch{2}.spm.util.imcalc.input(1) = cfg_dep('Realign: Estimate: Realigned Images (Sess 1)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','sess', '()',{1}, '.','cfiles'));
matlabbatch{2}.spm.util.imcalc.output = '';
matlabbatch{2}.spm.util.imcalc.outdir = {''};
matlabbatch{2}.spm.util.imcalc.expression = 'mean(X)';
matlabbatch{2}.spm.util.imcalc.var = struct('name', {}, 'value', {});
matlabbatch{2}.spm.util.imcalc.options.dmtx = 1;
matlabbatch{2}.spm.util.imcalc.options.mask = 0;
matlabbatch{2}.spm.util.imcalc.options.interp = 1;
matlabbatch{2}.spm.util.imcalc.options.dtype = 4;
matlabbatch{3}.spm.spatial.preproc.channel.vols(1) = cfg_dep('Image Calculator: ImCalc Computed Image: ', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
matlabbatch{3}.spm.spatial.preproc.channel.biasreg = 0.001;
matlabbatch{3}.spm.spatial.preproc.channel.biasfwhm = 60;
matlabbatch{3}.spm.spatial.preproc.channel.write = [0 0];
matlabbatch{3}.spm.spatial.preproc.tissue(1).tpm = {'/DATA/doeringe/Dokumente/spm12/tpm/TPM.nii,1'};
matlabbatch{3}.spm.spatial.preproc.tissue(1).ngaus = 1;
matlabbatch{3}.spm.spatial.preproc.tissue(1).native = [1 0];
matlabbatch{3}.spm.spatial.preproc.tissue(1).warped = [0 0];
matlabbatch{3}.spm.spatial.preproc.tissue(2).tpm = {'/DATA/doeringe/Dokumente/spm12/tpm/TPM.nii,2'};
matlabbatch{3}.spm.spatial.preproc.tissue(2).ngaus = 1;
matlabbatch{3}.spm.spatial.preproc.tissue(2).native = [1 0];
matlabbatch{3}.spm.spatial.preproc.tissue(2).warped = [0 0];
matlabbatch{3}.spm.spatial.preproc.tissue(3).tpm = {'/DATA/doeringe/Dokumente/spm12/tpm/TPM.nii,3'};
matlabbatch{3}.spm.spatial.preproc.tissue(3).ngaus = 2;
matlabbatch{3}.spm.spatial.preproc.tissue(3).native = [1 0];
matlabbatch{3}.spm.spatial.preproc.tissue(3).warped = [0 0];
matlabbatch{3}.spm.spatial.preproc.warp.mrf = 1;
matlabbatch{3}.spm.spatial.preproc.warp.cleanup = 1;
matlabbatch{3}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
matlabbatch{3}.spm.spatial.preproc.warp.affreg = 'mni';
matlabbatch{3}.spm.spatial.preproc.warp.fwhm = 0;
matlabbatch{3}.spm.spatial.preproc.warp.samp = 3;
matlabbatch{3}.spm.spatial.preproc.warp.write = [0 0];
matlabbatch{3}.spm.spatial.preproc.warp.vox = NaN;
matlabbatch{3}.spm.spatial.preproc.warp.bb = [NaN NaN NaN
                                              NaN NaN NaN];
matlabbatch{4}.spm.util.imcalc.input(1) = cfg_dep('Image Calculator: ImCalc Computed Image: ', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
matlabbatch{4}.spm.util.imcalc.input(2) = cfg_dep('Segment: c1 Images', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{1}, '.','c', '()',{':'}));
matlabbatch{4}.spm.util.imcalc.input(3) = cfg_dep('Segment: c2 Images', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{2}, '.','c', '()',{':'}));
matlabbatch{4}.spm.util.imcalc.input(4) = cfg_dep('Segment: c3 Images', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','tiss', '()',{3}, '.','c', '()',{':'}));
matlabbatch{4}.spm.util.imcalc.output = '';
matlabbatch{4}.spm.util.imcalc.outdir = {'/DATA/doeringe/Dokumente/BrainAge/2_Segmented'};
matlabbatch{4}.spm.util.imcalc.expression = 'i1.*(i2+i3+i4)';
matlabbatch{4}.spm.util.imcalc.var = struct('name', {}, 'value', {});
matlabbatch{4}.spm.util.imcalc.options.dmtx = 0;
matlabbatch{4}.spm.util.imcalc.options.mask = 0;
matlabbatch{4}.spm.util.imcalc.options.interp = 1;
matlabbatch{4}.spm.util.imcalc.options.dtype = 4;
matlabbatch{5}.spm.spatial.coreg.estwrite.ref = {'/DATA/doeringe/Dokumente/spm12/tpm/TPM.nii,1'};
matlabbatch{5}.spm.spatial.coreg.estwrite.source(1) = cfg_dep('Image Calculator: ImCalc Computed Image: ', substruct('.','val', '{}',{4}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
matlabbatch{5}.spm.spatial.coreg.estwrite.other = {''};
matlabbatch{5}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';
matlabbatch{5}.spm.spatial.coreg.estwrite.eoptions.sep = [4 2];
matlabbatch{5}.spm.spatial.coreg.estwrite.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
matlabbatch{5}.spm.spatial.coreg.estwrite.eoptions.fwhm = [7 7];
matlabbatch{5}.spm.spatial.coreg.estwrite.roptions.interp = 4;
matlabbatch{5}.spm.spatial.coreg.estwrite.roptions.wrap = [0 0 0];
matlabbatch{5}.spm.spatial.coreg.estwrite.roptions.mask = 0;
matlabbatch{5}.spm.spatial.coreg.estwrite.roptions.prefix = 'r';
matlabbatch{6}.spm.spatial.normalise.estwrite.subj.vol(1) = cfg_dep('Coregister: Estimate & Reslice: Coregistered Images', substruct('.','val', '{}',{5}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','rfiles'));
matlabbatch{6}.spm.spatial.normalise.estwrite.subj.resample(1) = cfg_dep('Coregister: Estimate & Reslice: Coregistered Images', substruct('.','val', '{}',{5}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','rfiles'));
matlabbatch{6}.spm.spatial.normalise.estwrite.eoptions.biasreg = 0.0001;
matlabbatch{6}.spm.spatial.normalise.estwrite.eoptions.biasfwhm = 60;
matlabbatch{6}.spm.spatial.normalise.estwrite.eoptions.tpm = {'/DATA/doeringe/Dokumente/spm12/tpm/TPM.nii'};
matlabbatch{6}.spm.spatial.normalise.estwrite.eoptions.affreg = 'mni';
matlabbatch{6}.spm.spatial.normalise.estwrite.eoptions.reg = [0 0.001 0.5 0.05 0.2];
matlabbatch{6}.spm.spatial.normalise.estwrite.eoptions.fwhm = 0;
matlabbatch{6}.spm.spatial.normalise.estwrite.eoptions.samp = 3;
matlabbatch{6}.spm.spatial.normalise.estwrite.woptions.bb = [-78 -112 -70
                                                             78 76 85];
matlabbatch{6}.spm.spatial.normalise.estwrite.woptions.vox = [2 2 2];
matlabbatch{6}.spm.spatial.normalise.estwrite.woptions.interp = 4;
matlabbatch{6}.spm.spatial.normalise.estwrite.woptions.prefix = 'w';
