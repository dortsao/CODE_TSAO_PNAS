leaf_dir = 'synthetic_bear'; %%%synthetic leaves
for frame = 1:4
      fname = sprintf('C:\\Users\\Doris\\Dropbox\\PAPERS_WORKING\\space\\code\\codenew_103020\\%s\\FI_%d.mat', leaf_dir, frame);
    load(fname);
  FrameInfo{frame}.tse =  twosided_edgeppoints;
end
  