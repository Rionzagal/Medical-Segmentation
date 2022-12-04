%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Test code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = struct

%% Loading CT images:
for k = 49:58
    name = string(compose('./ct_scans/0%d.nii',k));
    data(k-48).ct_scans = niftiread(name);
end

for k = 66:99
    name = string(compose('./ct_scans/0%d.nii',k));
    data(k-55).ct_scans = niftiread(name);
end

for k = 100:130
    name = string(compose('./ct_scans/%d.nii',k));
    data(k-55).ct_scans = niftiread(name);
end

%% Loading stroke segmentations:
for k = 49:58
    name = string(compose('./masks/0%d.nii',k));
    data(k-48).masks = niftiread(name);
end

for k = 66:99
    name = string(compose('./masks/0%d.nii',k));
    data(k-55).masks = niftiread(name);
end

for k = 100:130
    name = string(compose('./masks/%d.nii',k));
    data(k-55).masks = niftiread(name);
end

%% Visualizing the scans (example):
sliceViewer(data(1).ct_scans)
max(data(1).ct_scans)
%% Visualizing the masks (example):
sliceViewer(data(1).masks)
