function accuracy = matchID(predictedLabel, targetLabel)

correctP=0;
for i=1:size(targetLabel,1)
    if strcmp(predictedLabel(i,:),targetLabel(i,:))
        correctP=correctP+1;
    end
end
accuracy = correctP/size(targetLabel,1)*100;

end