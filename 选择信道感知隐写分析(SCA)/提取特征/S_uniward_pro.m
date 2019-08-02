function weight=S_uniward_pro(img,payload)

sigma_bar = 10*eps;     % testing stabilization constant
img=double(img);
[~, weight] = S_UNIWARD(img, payload, sigma_bar);




