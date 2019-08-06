export const state = () => ({
  data: [
    { id: 1, name: 'Hands on head', image: require('~/assets/images/poses/hands-on-head.png') },
    { id: 2, name: 'Raise left hand', image: require('~/assets/images/poses/victory.png') },
    { id: 3, name: 'Spread hands', image: require('~/assets/images/poses/cheer-up.png') },
    { id: 4, name: 'Point to right', image: require('~/assets/images/poses/go-next.png') },
    { id: 5, name: 'Point to left', image: require('~/assets/images/poses/go-back.png') },
    { id: 6, name: 'Ultraman', image: require('~/assets/images/poses/ultraman.png') },
    { id: 7, image: '' },
  ]
})

export const mutations = {
  remove(state, payload) {
    state.data = state.data.filter(pose => pose.id !== payload.id)
  }
}

export const actions = {
  remove({ commit }, { poseId }) {
    commit('remove', { id: poseId })
  }
}
