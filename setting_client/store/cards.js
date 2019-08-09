export const state = () => ({
  data: [
    { id: 1, name: 'move-next', actionType: "slide", image: require('~/assets/images/actions/move-next.png'), poseId: 7},
    { id: 2, name: 'move-prev', actionType: "slide", image: require('~/assets/images/actions/move-prev.png'), poseId: 7},
    { id: 3, name: 'yeah', actionType: "sound", image: require('~/assets/images/actions/yeah.png'), poseId: 7},
    { id: 4, name: 'patipati', actionType: "sound", image: require('~/assets/images/actions/patipati.png'), poseId: 7},
    { id: 5, name: 'gaan', actionType: "sound", image: require('~/assets/images/actions/gaan.png'), poseId: 7},
    { id: 6, name: 'yaru', actionType: "sound", image: require('~/assets/images/actions/yaru.png'), poseId: 7},
    { id: 7, name: 'www', actionType: "sound", image: require('~/assets/images/actions/www.png'), poseId: 7},
    { id: 8, name: 'gua', actionType: "sound", image: require('~/assets/images/actions/gua.png'), poseId: 7},
    { id: 9, name: 'shakin', actionType: "sound", image: require('~/assets/images/actions/shakin.png'), poseId: 7},
    { id: 10, name: 'trumpet', actionType: "sound", image: require('~/assets/images/actions/trumpet.png'), poseId: 7},
    { id: 11, name: 'tekagenn', actionType: "sound", image: require('~/assets/images/actions/tekagenn.png'), poseId: 7},
    { id: 12, name: 'scream-woman', actionType: "sound", image: require('~/assets/images/actions/scream-woman.png'), poseId: 7},
    { id: 13, name: 'ougi', actionType: "sound", image: require('~/assets/images/actions/ougi.png'), poseId: 7},
    { id: 14, name: 'sakiwoisogu', actionType: "sound", image: require('~/assets/images/actions/sakiwoisogu.png'), poseId: 7},
    { id: 15, name: 'matte', actionType: "sound", image: require('~/assets/images/actions/matte.png'), poseId: 7},
  ]
})

export const getters = {
  cardsByPose: state => poseId => {
    return state.data.filter(card => card.poseId === poseId)
  }
}

export const mutations = {
  set(state, { cards, poseId }) {
    state.data = state.data.filter(card => card.poseId !== poseId)
    const newCards = cards.map(card => {
      return {
        ...card,
        poseId: poseId
      }
    })
    state.data = state.data.concat(newCards)
  }
}

export const actions = {
  set({ commit }, { cards, poseId }) {
    commit('set', { cards, poseId })
  }
}
