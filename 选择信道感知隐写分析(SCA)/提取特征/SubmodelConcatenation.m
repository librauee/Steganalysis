function fea = SubmodelConcatenation(F)
Ss = fieldnames(F);
indexTo = 0;
    for Sid = 1:length(Ss)
        Fsingle = eval(['F.' Ss{Sid}]);
        indexFrom = indexTo + 1;
        fprintf('   F.%s : %d x %d\n', Ss{Sid}, size(Fsingle, 1), size(Fsingle, 2));
        indexTo = indexFrom + size(Fsingle,2) - 1;
        fea(1,indexFrom:indexTo) = Fsingle;
    end
end

