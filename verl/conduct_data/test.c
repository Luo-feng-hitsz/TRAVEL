#include<stdio.h>
#include<stdlib.h>
int main()
{
    int n, *a, m, x, y, i;
    scanf("%d",&n);
    a = (int*)malloc(n*sizeof(int));
    for(i = 0; i < n; i++)
        scanf("%d",&a[i]);
    scanf("%d", &m);
    for(i = 1; i <= m; i++ )
    {
        scanf("%d%d",&x, &y);
        if(((x-1) != 0) || (x != n))
        {
            a[x-2] += y-1;
            a[x] += a[x-1]-y;
        }
        a[x-1] = 0;
    }
    for(i =0; i < n; i++)
        printf("%d\n",a[i]);
    return 0;
}