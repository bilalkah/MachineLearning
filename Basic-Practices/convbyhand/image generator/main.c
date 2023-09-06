#include <stdlib.h>
#include <stdio.h>

int main()
{

    FILE *fp;
    FILE *ptr;

    unsigned char harf[9][7];
    int i = 0, j = 0, k = 0;
    char file[13];
    fp = fopen("Filename.txt", "r");
    while (!feof(fp))
    {
        fread(&file, sizeof(char), 13, fp);
        file[12] = '\0';

        ptr = fopen(file, "rb");
        unsigned char abc;
        i = 0; 
        j = 0;
        while (!feof(ptr))
        {
            fread(&abc, sizeof(unsigned char), 1, ptr);
            if (abc == '0' || abc == '1')
            {
                harf[i][j] = abc;
                j++;
            }
            if (j == 7)
            {
                i++;
                j = 0;
            }
        }
        fclose(ptr);

        file[9] = 'p';
        file[10] = 'g';
        file[11] = 'm';
        ptr = fopen(file, "wb");
        fprintf(ptr, "P5\n");
        fprintf(ptr, "28 36\n");
        fprintf(ptr, "255\n");
        unsigned char temp[28];
        for (i = 0; i < 9; i++)
        {
            for (j = 0; j < 4; j++)
            {
                for (k = 0; k < 7; k++)
                {
                    temp[k * 4] = (harf[i][k] - 48) * 255;
                    temp[k * 4 + 1] = (harf[i][k] - 48) * 255;
                    temp[k * 4 + 2] = (harf[i][k] - 48) * 255;
                    temp[k * 4 + 3] = (harf[i][k] - 48) * 255;
                }
                fwrite(temp, sizeof(unsigned char), 28, ptr);
            }
        }

        fclose(ptr);
    }
    fclose(fp);

    return 0;
}
