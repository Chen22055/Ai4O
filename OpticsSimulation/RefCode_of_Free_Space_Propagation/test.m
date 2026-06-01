A = ones(3,3);   % 读取图像
B = ones(2,2);
[rowsA, colsA] = size(A);
[rowsB, colsB] = size(B);

% 计算扩展大小
padRows = rowsA + rowsB - 1;
padCols = colsA + colsB - 1;

% 对 A 和 B 进行零填充
A_pad = padarray(A, [padRows - rowsA, padCols - colsA], 'post');
B_pad = padarray(B, [padRows - rowsB, padCols - colsB], 'post');