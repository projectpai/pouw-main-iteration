#include <stdio.h>

#include <sodium.h>
#include <sodium/crypto_vrf_ietfdraft03.h>
#include <string.h>
#include <math.h>

unsigned long char_arr_to_long(unsigned const char *const src)
{
    unsigned long dest;
    memcpy(&dest, src, sizeof(dest));
    return dest;
}

double cosine_similarity(const double *vec1, const double *vec2, unsigned int len)
{
    double dot = 0.0, a = 0.0, b = 0.0 ;
    for(unsigned int i = 0u; i < len; ++i) {
        dot += vec1[i] * vec2[i] ;
        a += vec1[i] * vec1[i] ;
        b += vec2[i] * vec2[i] ;
    }
    return dot / (sqrt(a) * sqrt(b)) ;
}

int main(void)
{
    // initialize the sodium library
    if (sodium_init() == -1) {
        return 1;
    }

    // VARIABLES
    // known public key
    unsigned char public_key[crypto_vrf_ietfdraft03_PUBLICKEYBYTES];

    // secret key is generated from public key
    unsigned char secret_key[crypto_vrf_ietfdraft03_SECRETKEYBYTES];
    // proof is subsequently generated
    unsigned char proof[crypto_vrf_ietfdraft03_PROOFBYTES];

    // hash is generated from the proof
    unsigned char hash[crypto_vrf_ietfdraft03_OUTPUTBYTES];

    // output
    unsigned char output[crypto_vrf_ietfdraft03_OUTPUTBYTES];

    // the input to the VRF
    const unsigned char message[] = "miner9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";


    // MAIN WORKFLOW
    // build a secret key based on the given public key
    if(crypto_vrf_ietfdraft03_keypair(public_key, secret_key))
        return 1;

    // create proof based on secret key and a common known string/message
    if(crypto_vrf_ietfdraft03_prove(proof, secret_key, message, (unsigned long long)sizeof(message)))
        return 1;

    // create the hash
    if(crypto_vrf_proof_to_hash(hash, proof))
        return 1;

    unsigned long long max_hash = (unsigned long long) pow(2, 64)-1;
    unsigned long long hash_val = char_arr_to_long(hash);
    long double hash_ratio = (long double)hash_val / (long double) max_hash;

    // ticket preferences vector
    const double ticket_prefs[] = {0, 7, 100};

    // corresponding task properties vector
    const double task_props[] = {0, 2, 1};

    double sim = cosine_similarity(ticket_prefs, task_props, 3);
    printf("Ratio: %.2Lf\n", hash_ratio);
    printf("Similariy: %.2f\n", sim);
    if(sim >= hash_ratio)
        printf("Ticket selected to work.\n");
    else
        printf("Ticket should wait for another task.\n");

    // verification (we need the public key, the proof and the message)
    if(crypto_vrf_ietfdraft03_verify(output, public_key, proof, message, (unsigned long long)sizeof(message))) {
        printf("Verification failed.");
        return 1;
    }

    printf("Verification succeeded.");
}