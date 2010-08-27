/* Public domain, FWIW
 * gcc -o genprimes -lgmp genprimes.c; ./genprimes > primes.bin
 * see http://www.ast.cam.ac.uk/~stg20/cuda/random/index.html
 */
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <gmp.h>

/* To verify the primes against the linked URL, compile this instead:
int main() {
    FILE *fp = fopen("primes.bin", "r");
    char stuff[5];
    while (fread(stuff, 4, 1, fp) == 1) {
        uint64_t i = *((uint32_t*)stuff);
        i = (i>>24) + (1<<8)*((i>>16)&0xff) + (1<<16)*((i>>8)&0xff) + (1<<24)*(i&0xff);
        uint64_t j = i * 4294967296L - 1;
        uint64_t k = (j-1)/2;
        printf("%lu %lu %lu\n", i, j, k);
    }
}
*/

int main(int argc, char* argv[]) {
    fprintf(stderr, "Generating list of multipliers for mod(2^32) MWC RNG\n");
    mpz_t candidate, twotothethirtytwo;
    mpz_init(candidate);
    mpz_init_set_d(twotothethirtytwo, (double) (4294967296L));

    char bytes[5];
    bytes[4] = 0;
    unsigned int i, found=0;
    for (i = 4294967295L; i > 2147483648; i--) {
        mpz_set_ui(candidate, i);
        mpz_mul(candidate, candidate, twotothethirtytwo);
        mpz_sub_ui(candidate, candidate, 1);
        if(mpz_probab_prime_p(candidate, 200)) {
            mpz_sub_ui(candidate, candidate, 1);
            mpz_tdiv_q_ui(candidate, candidate, 2);
            if(mpz_probab_prime_p(candidate, 200)) {
                bytes[0] = (i>>24)&0xff;
                bytes[1] = (i>>16)&0xff;
                bytes[2] = (i>>8)&0xff;
                bytes[3] = i&0xff;
                fwrite(bytes, 4, 1, stdout);
                found++;
                if (!(found&0xff)) fprintf(stderr, ".");
            }
        }
    }
    fprintf(stderr, "\nFound %d multipliers.\n", found);
    mpz_clear(candidate);
    mpz_clear(twotothethirtytwo);
}


