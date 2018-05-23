import api from '@/api.js'
import constPaths from '@/constants/constant-paths.js'

const state = {
  statements: [],
  loadingStatements: true,
  nextUrl: constPaths.MEDICATION_STATEMENT_URL
}

const getters = {
  statements: state => state.statements,
  loadingStatements: state => state.loadingStatements
}

const actions = {
  getStatements ({ state, commit }) {
    return api.fetch(state.nextUrl)
    .then(data => {
      commit('setList', data)
      return state.statements
    })
    .catch(error => Promise.reject(error))
  },
  clear ({ state, commit }) {
    commit('clear')
    return state.statements
  }
}

const mutations = {
  setList (state, data) {
    state.statements = [...state.statements, ...data.entry]
    if (data.link[1] && data.link[1].relation === 'next') {
      state.nextUrl = data.link[1].url
      state.loadingStatements = true
    } else {
      state.nextUrl = constPaths.MEDICATION_STATEMENT_URL
      state.loadingStatements = false
    }
  },
  clear (state) {
    state.statements = []
    state.nextUrl = constPaths.MEDICATION_STATEMENT_URL
    state.loadingStatements = true
  }
}

export default {
  namespaced: true,
  state,
  getters,
  actions,
  mutations
}
