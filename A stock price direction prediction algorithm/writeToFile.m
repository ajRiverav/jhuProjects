function writeToFile( string, filename, workerId )

if nargin<3
    workerId='';
end

fid = fopen([num2str(workerId) filename],'a');
fprintf(fid, string);
fclose(fid);

end

