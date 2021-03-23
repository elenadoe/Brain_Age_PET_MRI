dirFDG = '/DATA/doeringe/Dokumente/BrainAge/FDG_PET/';
dirToScripts = '/DATA/doeringe/Dokumente/Scripts (1)/CoregToFirstFrame/';

petFilesFDG = dir(fullfile(dirFDG,'s*.nii'));

for i = 1:numel(petFilesFDG)
    spm_jobman('initcfg');
    
    currentFDGFile = petFilesFDG(i).name;
       
    fin = fopen(strcat(dirToScripts,'CoregToFirstFrame_template.m'), 'r');
    fout = fopen(strcat(dirToScripts,'CoregToFirstFrame_intermed.m'), 'w');
    

    findstr1 = 'REPLACE_FDG_IMAGE';
    replacestr1 = strcat(dirFDG, currentFDGFile);
    
    % check if image has 52 frames
    
   
    
    while ~feof(fin)
        s = fgetl(fin);
        s = strrep(s, findstr1, replacestr1);
        fprintf(fout,'%s\n',s)
    end

    
    fclose(fin)
    fclose(fout)
    
    
    disp(currentFDGFile);
    disp('-----')
    
    
    %CODE FOR MATLAB BATCH PROCESS
        % List of open inputs
        nrun = 1; % enter the number of runs here

        jobfile = {strcat(dirToScripts,'CoregToFirstFrame_intermed.m')};

        
        spm('defaults', 'PET');
        spm_jobman('run', jobfile);
    %END CODE FOR MATLAB BATCH PROCESS
    
end
