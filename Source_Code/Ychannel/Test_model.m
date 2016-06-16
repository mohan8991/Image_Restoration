function [] = Test_model(im,model,up_scale,iter,img_name)
% =========================================================================
% Test code build for CAP 6412 cost Project based on 
% Super-Resolution Convolutional Neural Networks (SRCNN)
% Syed Ahmed.
% -------------------------------------------------------------------------
% Reference
%   Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. Learning a  
%   Deep Convolutional Network for Image Super-Resolution,in Proceedings of 
%	European Conference on Computer Vision (ECCV), 2014
%   Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. 
%   Image Super-Resolution Using Deep Convolutional Networks,  
%   arXiv:1501.00092
% -------------------------------------------------------------------------
% Function specs: im - the test image; model- the model to be test in .mat format
% iter - no. of iterations the model is trained on.
% img_name - name of the image to avoid confusion while storing the evaluation results.
% =========================================================================


%% Select the Y channel alnoe since the model is trained for the same....
if size(im,3)>1
    im = rgb2ycbcr(im);
    im = im(:, :, 1);
end
im_gnd = modcrop(im, up_scale);
im_gnd = single(im_gnd)/255;

%% bicubic interpolation
im_l = imresize(im_gnd, 1/up_scale, 'bicubic');
im_b = imresize(im_l, up_scale, 'bicubic');

%% SRCNN
im_h = SRCNN(model, im_b);

%% remove border
im_h = shave(uint8(im_h * 255), [up_scale, up_scale]);
im_gnd = shave(uint8(im_gnd * 255), [up_scale, up_scale]);
im_b = shave(uint8(im_b * 255), [up_scale, up_scale]);

%% compute PSNR
psnr_bic = compute_psnr(im_gnd,im_b);
psnr_srcnn = compute_psnr(im_gnd,im_h);

%% show results
fprintf('PSNR for Bicubic Interpolation: %f dB\n', psnr_bic);
fprintf('PSNR for SRCNN Reconstruction: %f dB\n', psnr_srcnn);

%% Save results
fid=fopen('Evaluation_Results.txt','a+');
fprintf(fid, 'Image name : %s \n', img_name);
fprintf(fid, 'For %d backward propagations and an upscale of %d, the evaluation results are: \n', iter, up_scale);
fprintf(fid, 'PSNR for Bicubic Interpolation: %f dB \n', psnr_bic);
fprintf(fid, 'PSNR for SRCNN Reconstruction: %f dB \n \n', psnr_srcnn);
fclose(fid);
%% show output image
%figure, imshow(im_b); title('Bicubic Interpolation');
%figure, imshow(im_h); title('SRCNN Reconstruction');
%% save output image
imwrite(im_b, [sprintf('%s_Bicubic_Interpolation_%d_%d',img_name,iter,up_scale) '.bmp']);% Saving the output for bicubic
imwrite(im_h, [sprintf('%s_SRCNN_Reconstruction_%d_%d',img_name,iter,up_scale) '.bmp']); % Saving the output for SRCNN