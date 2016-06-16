% =========================================================================
% Test code build for CAP 6412 cost Project based on 
% Super-Resolution Convolutional Neural Networks (SRCNN)
% Syed Ahmed.
% -------------------------------------------------------------------------
% The below code is used to automatically generate the evaluation results for
% all the test images on the different trained models.
% After the code is been run, all the output images are stored in the current folder 
% in the following format:
% 'paradigm' '# backprops the model has been trained for'. 'extension'
% Also, the evaluation results are stored in the text file name 'Evaluation_results.txt'
% =========================================================================

close all;
clear all;
folder_mod  = 'models/';
folder_im = 'Set14/';
FolderInfo_mod = dir(folder_mod);
FolderInfo_im  = dir(folder_im);
for i=3 : length(FolderInfo_im) % testig for all images...
	for j=3 : length(FolderInfo_mod) % testing for each model...
		for k=2 : 3 % testing for varied scale...
			im_name = [folder_im FolderInfo_im(i).name];
			im = imread(im_name);
			im_name = im_name(7:11);
			model = [folder_mod FolderInfo_mod(j).name];
			iter = sscanf(model, '%*[^0123456789]%d');
			up_scale = k;
			Test_model(im,model,up_scale,iter,im_name)
		end
	end
end
