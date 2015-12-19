#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/CrossInputNeighborhood.c"
#else

#include "stdio.h"

static int nn_(CrossInputNeighborhood_updateOutput)(lua_State *L)
{
	printf("inside updateOutput nn \n ");

	return 1;
}

static int nn_(CrossInputNeighborhood_updateGradInput)(lua_State *L)
{
	printf("inside updateGradInput nn");

	return 1;
}

static const struct luaL_Reg nn_(CrossInputNeighborhood__) [] = {
	{	"CrossInputNeighborhood_updateOutput", nn_(CrossInputNeighborhood_updateOutput)},
	{	"CrossInputNeighborhood_updateGradInput", nn_(CrossInputNeighborhood_updateGradInput)},
	{	NULL, NULL}
};

static void nn_(CrossInputNeighborhood_init)(lua_State *L)
{
	luaT_pushmetatable(L, torch_Tensor);
	luaT_registeratname(L, nn_(CrossInputNeighborhood__), "nn");
	lua_pop(L,1);
}


#endif
