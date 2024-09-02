% 将nan值替换为-9999
data0= readtable('output_data.csv');
data1 = table2array(data0);
% data1(isnan(data1)) = -9999;
x = data1(:, 1);
y = data1(:, 2);
z = data1(:, 3);
water = data1(:, 4);
rock = data1(:, 5);
val=rock;


% % 创建坐标网格
[xi, yi, zi] = meshgrid(unique(x), unique(y), unique(z));


% % 用 scatteredInterpolant 函数将数据插值到均匀网格中
F = scatteredInterpolant(x, y, z, val);
V = F(xi, yi, zi);

new_V=V;
stlname='rock2.stl';
% count = sum(water(:) == 3);
new_V(new_V == 2) = 5;
% V(V ~= 2) = 0;
% V(V == 1) = 5;
% V(V == 2) = 5;
isovalue=5;
% 绘制平滑处理前的模型
[faces, verts] = extractIsosurface(new_V,isovalue);
figure
p = patch('Faces',faces,'Vertices',verts);
isonormals(new_V,p)
view(3)
set(p,'FaceColor',[0.5 1 0.5])
set(p,'EdgeColor','none')
camlight
lighting gouraud
title('Before Smoothing')

% T_before = triangulation(double(faces),double(verts));
% stlwrite(T_before, 'model_before.stl')

% % 对模型进行局部平均平滑
% verts_new = verts;
% for i = 1:size(verts,1)
%     [row, ~] = find(faces == i);
%     neighbors = unique(faces(row,:));
%     neighbors = neighbors(neighbors ~= i);  % 删除顶点本身
%     if ~isempty(neighbors)  % 只有当存在邻居时才计算平均位置
%         verts_new(i,:) = mean(verts(neighbors,:));
%     end
% end

%高斯滤波
% V_smooth = imgaussfilt3(V, 1);  % 第二个参数是sigma，控制平滑程度
V_smooth = medfilt3(new_V, [5 5 5]);  % 第二个参数是sigma，控制平滑程度

% % 定义结构元素
% se = strel('sphere', 4);
% % 执行开运算：先腐蚀后膨胀
% open_V = imdilate(imerode(V_smooth, se), se);
% % % 执行闭运算：先膨胀后腐蚀
% close_V = imerode(imdilate(open_V, se), se);

[facesf, vertsf] = extractIsosurface(V_smooth,isovalue);
% 平滑处理
verts_smooth = vertsf;

% 绘制平滑处理后的模型
figure
p = patch('Faces',facesf,'Vertices',verts_smooth);
isonormals(new_V,p)
view(3)
set(p,'FaceColor',[0.5 1 0.5])
set(p,'EdgeColor','none')
camlight
lighting gouraud
title('After Smoothing')

T_after = triangulation(double(facesf), double(verts_smooth));

% % 生成 Alpha Shape
% xx = verts_smooth(:, 1);
% yy = verts_smooth(:, 2);
% zz = verts_smooth(:, 3);
% xx = double(xx);
% yy = double(yy);
% zz = double(zz);
% % 生成 Alpha Shape
% for alpha_val = [1, 2, 3, 4]  % 不同的alpha值
%     shp = alphaShape(xx, yy, zz, alpha_val);
%     figure;
%     plot(shp);
%     title(['Alpha Shape with alpha = ', num2str(alpha_val)]);
%     [bf, P] = boundaryFacets(shp);
%     tri_obj = triangulation(bf, P);
%     stl_filename = sprintf('alpha_shape_%.2f.stl', alpha_val);
%     stlwrite(tri_obj, stl_filename);
%     fprintf('STL file for alpha_val=%.2f has been saved as %s\n', alpha_val, stl_filename);
% end

stlwrite(T_after, stlname)    

% 展示平滑前后的三角形网格图像
T_before = triangulation(double(faces),double(verts));
% 删除verts_new中的无效行
verts_smooth(any(~isfinite(verts_smooth), 2), :) = [];
% 创建三角剖分对象
T_after = triangulation(double(facesf), double(verts_smooth));

figure
subplot(1,2,1);
trimesh(T_before)
title('Before Smoothing')

subplot(1,2,2);
trimesh(T_after)
title('After Smoothing')

