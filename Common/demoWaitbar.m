
f = waitbar(0,sprintf("nothing\n \n "),'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
setappdata(f,'canceling',0);
for i = 1:10
    
    waitbar(i/10,f,sprintf("something\n \n "));
    
    if getappdata(f,'canceling')
        delete(f);
        return
    end
    
    pause(1);

end

disp("code below");

delete(f);
