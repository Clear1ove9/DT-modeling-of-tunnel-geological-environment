data0= readtable('output_data.csv');
data1 = table2array(data0);

% 提取坐标和温度数据
xyz_data = data1(:, 1:3);
temp_data = data1(:, 5);

% 网格尺寸m
m = 2;
% 仅绘制温度为m的网格
target_temp = 2;
stlname='rock_cube2.stl';
% 计算数据集空间范围
x_min = min(xyz_data(:, 1));
x_max = max(xyz_data(:, 1));
y_min = min(xyz_data(:, 2));
y_max = max(xyz_data(:, 2));
z_min = min(xyz_data(:, 3));
z_max = max(xyz_data(:, 3));

% 初始化网格
x_range = x_min:m:x_max;
y_range = y_min:m:y_max;
z_range = z_min:m:z_max;
grid_temp = zeros(length(x_range), length(y_range), length(z_range));

% 将每个点映射到相应的网格并计算“代表温度”
n = length(temp_data);
for i = 1:n
    x_idx = floor((xyz_data(i, 1) - x_min) / m) + 1;
    y_idx = floor((xyz_data(i, 2) - y_min) / m) + 1;
    z_idx = floor((xyz_data(i, 3) - z_min) / m) + 1;
    temp = temp_data(i);
    
    if grid_temp(x_idx, y_idx, z_idx) == 0
        grid_temp(x_idx, y_idx, z_idx) = temp;
    else
        grid_temp(x_idx, y_idx, z_idx) = max(grid_temp(x_idx, y_idx, z_idx), temp);
    end
end

% 可视化
figure;
view(3); % 设置三维视图

for x_idx = 1:length(x_range)
    for y_idx = 1:length(y_range)
        for z_idx = 1:length(z_range)
            temp = grid_temp(x_idx, y_idx, z_idx);
            
            if temp == target_temp
                vertices = [x_range(x_idx), y_range(y_idx), z_range(z_idx);
                            x_range(x_idx) + m, y_range(y_idx), z_range(z_idx);
                            x_range(x_idx) + m, y_range(y_idx) + m, z_range(z_idx);
                            x_range(x_idx), y_range(y_idx) + m, z_range(z_idx);
                            x_range(x_idx), y_range(y_idx), z_range(z_idx) + m;
                            x_range(x_idx) + m, y_range(y_idx), z_range(z_idx) + m;
                            x_range(x_idx) + m, y_range(y_idx) + m, z_range(z_idx) + m;
                            x_range(x_idx), y_range(y_idx) + m, z_range(z_idx) + m];
                
                faces = [1, 2, 3, 4;
                         2, 6, 7, 3;
                         3, 7, 8, 4;
                         4, 8, 5, 1;
                         1, 5, 6, 2;
                         5, 8, 7, 6];
                
                patch('Faces', faces, 'Vertices', vertices, 'FaceColor', [0, 0, temp / 5], 'FaceAlpha', 0.5);
            end
        end
    end
end

% 坐标轴标签和颜色条
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Grid-based Temperature Visualization');
colorbar;

% 初始化用于存储所有面和顶点的矩阵
all_faces = [];
all_vertices = [];

vertex_count = 1;  % 用于跟踪顶点数量

% 与之前一样遍历所有网格
for x_idx = 1:length(x_range)
    for y_idx = 1:length(y_range)
        for z_idx = 1:length(z_range)
            temp = grid_temp(x_idx, y_idx, z_idx);
            
            if temp == target_temp
                vertices = [x_range(x_idx), y_range(y_idx), z_range(z_idx);
                            x_range(x_idx) + m, y_range(y_idx), z_range(z_idx);
                            x_range(x_idx) + m, y_range(y_idx) + m, z_range(z_idx);
                            x_range(x_idx), y_range(y_idx) + m, z_range(z_idx);
                            x_range(x_idx), y_range(y_idx), z_range(z_idx) + m;
                            x_range(x_idx) + m, y_range(y_idx), z_range(z_idx) + m;
                            x_range(x_idx) + m, y_range(y_idx) + m, z_range(z_idx) + m;
                            x_range(x_idx), y_range(y_idx) + m, z_range(z_idx) + m];
                
                % 将每个四边形面分割为两个三角形面
                quads = [1, 2, 3, 4;
                         2, 6, 7, 3;
                         3, 7, 8, 4;
                         4, 8, 5, 1;
                         1, 5, 6, 2;
                         5, 8, 7, 6];
                
                triangles = [quads(:, 1), quads(:, 2), quads(:, 3);
                             quads(:, 1), quads(:, 3), quads(:, 4)] + vertex_count - 1;  % 更新三角形的索引
                
                % 将新的顶点和面添加到总矩阵中
                all_vertices = [all_vertices; vertices];
                all_faces = [all_faces; triangles];
                
                % 更新顶点计数器
                vertex_count = vertex_count + size(vertices, 1);
            end
        end
    end
end

% 使用三角面代替四边形面，导出为STL格式
T_after = triangulation(double(all_faces), double(all_vertices));


% 提取所有的边
edges = edges(T_after);

% 查找只出现一次（即属于外表面）的边
[~, ~, ic] = unique(sort(edges, 2), 'rows');
edgeCounts = accumarray(ic, 1);
exteriorEdges = edges(edgeCounts == 1, :);

% 查找与这些边相关联的面
attachedFacesCellArray = edgeAttachments(T_after, exteriorEdges);

% 预分配数组
exteriorFaces = zeros(size(all_faces, 1), 1);
count = 0;

for i = 1:length(attachedFacesCellArray)
    numFaces = length(attachedFacesCellArray{i});
    exteriorFaces(count+1:count+numFaces) = attachedFacesCellArray{i};
    count = count + numFaces;
end

% 裁剪数组以去除多余的空间
exteriorFaces = exteriorFaces(1:count);

% 去除重复项
exteriorFaces = unique(exteriorFaces);

newTri = triangulation(all_faces(exteriorFaces, :), all_vertices);

% 写入新的STL文件
stlwrite(newTri,stlname);
% stlwrite(T_after, filename)   