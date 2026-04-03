close all
clear
tic

fname = 'subject_passive.json';
fid = fopen(fname); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid); 
data = jsondecode(str);
%% getting the times of event and values at that time
fields=fieldnames(data); %contains the time of event
values=struct2cell(data); % contains all the data for event (i.e., score, hit, etc)

%% grabbing the values
i=0;
k=0;
indexA=[];
indexB=[];
for i=1:length(values)
        strValues=split(values(i),',');
        scoreStr=split(strValues(1),':');
        score(i,1)=str2double(scoreStr(2));
        healthStr=split(strValues(2),':');
        health(i,1)=str2double(healthStr(2));
        currentTimeStr = extractBetween(strValues(3),27,38);
        time(i,1:3)=str2double(split(currentTimeStr,':'))';
        yPosStr=split(strValues(4),':');
        yPos(i,1)=str2double(yPosStr(2));
        upKeyStr=split(strValues(5),':');
            keyPressed=matches(upKeyStr,'true');
        upKey(i,1)=keyPressed(2,1);
        downKeyStr=split(strValues(6),':');
            keyPressedDown=matches(downKeyStr,'true');
        downKey(i,1)=keyPressedDown(2,1);
        asteroidsPresentStr=split(strValues(7),':');
            asteroidsYN=matches(asteroidsPresentStr,'true');
        asteroids(i,1)=asteroidsYN(2,1);
        trialStr=split(strValues(8),':');
        trial(i,1)=str2double(trialStr(2));
        try
            aSafeStr=split(strValues(9),':');
            aSafe(i,1)=str2double(aSafeStr(2));
        catch
            indexA = [indexA; i];
            aSafe(i,1)=NaN;
        end    
        try
            bSafe(i,1)=str2double(extractBetween(strValues(10),':','}'));           
        catch
            indexB = [indexB; i];
            bSafe(i,1)=NaN;
        end
end

%% Populate everything in a cell
allDataJson={};
allDataJson{1,1}='score';
allDataJson{2,1}=score;
allDataJson{1,2}='health';
allDataJson{2,2}=health;
allDataJson{1,3}='time';
allDataJson{2,3}=time;
allDataJson{1,4}='yPosition';
allDataJson{2,4}=yPos;
allDataJson{1,5}='upKeyPress';
allDataJson{2,5}=upKey;
allDataJson{1,6}='downKeyPress';
allDataJson{2,6}=downKey;
allDataJson{1,7}='asteroidsPresent';
allDataJson{2,7}=asteroids;
allDataJson{1,8}='trialNumber';
allDataJson{2,8}=trial;
allDataJson{1,9}='A_safe';
allDataJson{2,9}=aSafe;
allDataJson{1,10}='B_safe';
allDataJson{2,10}=bSafe;
%%
allDataTask=allDataJson;
 save ('allDataTask.mat', 'allDataTask')
toc